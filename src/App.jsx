import { useState, useEffect, useRef } from "react";
import "./App.css";

const PATCH_COLORS = [
  "#ff6b6b", "#ffa94d", "#ffd43b", "#69db7c", "#38d9a9",
  "#4dabf7", "#748ffc", "#da77f2", "#f783ac", "#a9e34b",
  "#63e6be", "#74c0fc", "#e599f7", "#ffa8a8", "#ffec99",
  "#b2f2bb", "#a5d8ff", "#d0bfff", "#fcc2d7", "#c3fae8"
];

const isMasked = (i) => {
  return maskedSet.has(i);
};

const maskedSet = new Set();
for (let i = 0; i < 196; i++) {
  if ((i % 3 !== 0) && (i % 7 !== 4) && (i % 11 !== 3)) maskedSet.add(i);
}

const STEPS = [
  {
    id: "rgb",
    label: "RGB Input",
    phase: "INPUT",
    color: "#f87171",
    desc: "A 224×224 colour image arrives as 3 stacked grids — Red, Green, Blue. Each pixel holds one brightness value per channel, giving us 3 × 224 × 224 = 150,528 raw numbers.",
    shape: "(B, 3, 224, 224)",
    tip: "Each channel is a grayscale heatmap of one colour's intensity.",
  },
  {
    id: "patches",
    label: "Patch Extraction",
    phase: "ENCODER",
    color: "#fb923c",
    desc: "A Conv2d with kernel_size=16 and stride=16 slices the image into a 14×14 grid of non-overlapping patches. Each patch covers 16×16 pixels across all 3 channels — 768 raw values per patch.",
    shape: "(B, 768, 14, 14)",
    tip: "kernel = stride means zero overlap — every pixel belongs to exactly one patch.",
  },
  {
    id: "embed",
    label: "Patch Embedding",
    phase: "ENCODER",
    color: "#facc15",
    desc: "768 learned filters compress each patch's 768 raw values into a 768-dim embedding vector. The spatial grid is flattened and transposed, producing 196 tokens — just like word tokens in a language model.",
    shape: "(B, 196, 768)",
    tip: "This linear projection is what Conv2d actually does — a learned dot product per patch.",
  },
  {
    id: "mask",
    label: "MAE Masking",
    phase: "ENCODER",
    color: "#4ade80",
    desc: "MAE randomly discards ~75% of patch tokens before encoding. This aggressive masking forces the model to learn semantic representations rather than low-level shortcuts — it can't just copy neighbouring pixels.",
    shape: "(B, ~49, 768)",
    tip: "75% masking ratio was found optimal in the original MAE paper by He et al., 2021.",
  },
  {
    id: "posenc",
    label: "Positional Encoding",
    phase: "ENCODER",
    color: "#34d399",
    desc: "Since Transformers are permutation-invariant, a fixed 2D sinusoidal positional embedding is element-wise added to each visible token. Using sin(pos/10000^{2i/d}) and cos(pos/10000^{2i/d}) for each (row, col), this injects spatial structure without adding learnable parameters.",
    shape: "(B, ~49, 768)",
    tip: "MAE uses fixed sin-cos embeddings (not learned). Each dimension encodes position at a different frequency — low dims capture coarse location, high dims capture fine position.",
  },
  {
    id: "encoder",
    label: "Transformer Encoder",
    phase: "ENCODER",
    color: "#22d3ee",
    desc: "The ~49 visible tokens pass through 12 Transformer blocks (ViT-Base). Each block applies: LayerNorm → Multi-Head Self-Attention (12 heads, dim_head=64) → Residual Add → LayerNorm → FFN (768→3072→768, GELU) → Residual Add. Only visible tokens are processed — 4× fewer than full sequence.",
    shape: "(B, ~49, 768)",
    tip: "ViT-Base config: 12 blocks, 12 attention heads, embed_dim=768, MLP ratio=4. Processing only ~25% of tokens gives a quadratic speedup in self-attention (O(n²) complexity).",
  },
  {
    id: "enc_projection",
    label: "Encoder → Decoder Projection",
    phase: "DECODER",
    color: "#6366f1",
    desc: "A linear layer projects encoder output from encoder_dim=768 to decoder_dim=512. This dimensionality reduction is essential — it decouples the encoder and decoder widths, letting the decoder be significantly narrower and more compute-efficient.",
    shape: "(B, ~49, 512)",
    tip: "Without this projection, the decoder would need to match the encoder's width (768). The 768→512 linear layer has only 393K parameters — negligible overhead for major flexibility.",
  },
  {
    id: "decode_tokens",
    label: "Decoder Token Assembly",
    phase: "DECODER",
    color: "#818cf8",
    desc: "The projected encoder output (~49 visible tokens, now dim=512) is combined with learned [MASK] tokens — a single shared 512-dim vector replicated to all ~147 masked positions. Tokens are re-ordered by their original patch index to restore the full 196-token spatial sequence.",
    shape: "(B, 196, 512)",
    tip: "All ~147 masked positions share the same learned [MASK] embedding. They become distinguishable only after positional encoding is added — position is their sole identity signal.",
  },
  {
    id: "decode_pos",
    label: "Decoder Positional Enc.",
    phase: "DECODER",
    color: "#a78bfa",
    desc: "A separate set of fixed 2D sin-cos positional encodings is added to all 196 tokens. This is critical — [MASK] tokens arrive with identical embeddings, so positional encoding is the only signal telling the decoder where each masked patch belongs in the image grid.",
    shape: "(B, 196, 512)",
    tip: "The decoder uses its own positional embeddings, independent of the encoder's. These are also fixed sin-cos (not learned), computed over the full 14×14 grid.",
  },
  {
    id: "decoder",
    label: "Transformer Decoder",
    phase: "DECODER",
    color: "#c084fc",
    desc: "8 lightweight Transformer blocks process all 196 tokens via self-attention (not cross-attention). Each block: LayerNorm → MHSA (16 heads, dim_head=32) → Residual Add → LayerNorm → FFN (512→2048→512, GELU) → Residual Add. Visible and mask tokens attend to each other, allowing mask tokens to gather reconstruction context.",
    shape: "(B, 196, 512)",
    tip: "The decoder is intentionally asymmetric — 8 blocks vs encoder's 12, dim=512 vs 768. This asymmetry is key: the encoder does the heavy semantic lifting, while the decoder only needs to solve the pixel-level reconstruction task.",
  },
  {
    id: "reconstruct",
    label: "Pixel Reconstruction",
    phase: "DECODER",
    color: "#f472b6",
    desc: "A final linear head projects each decoder token: 512 → 768 (= 16×16×3 pixel values). Loss = MSE between predicted and target pixels, computed only on masked patches. Crucially, target pixels are per-patch normalized (subtract patch mean, divide by patch std) — this shifts focus from low-frequency colour to high-frequency structure.",
    shape: "(B, 196, 768)",
    tip: "Per-patch normalization of targets is critical — without it, the model learns to predict the mean colour (trivial low-frequency solution). Normalized targets force learning of textures, edges, and semantic structure.",
  },
  {
    id: "output",
    label: "Reconstructed Image",
    phase: "OUTPUT",
    color: "#fb7185",
    desc: "The 196 predicted patch vectors are un-patchified — reshaped from (B, 196, 768) → (B, 3, 224, 224) by placing each 16×16×3 block back into its grid position. At inference, the decoder is discarded entirely — only the encoder is kept as a pre-trained feature backbone for downstream tasks.",
    shape: "(B, 3, 224, 224)",
    tip: "After pre-training, the encoder is fine-tuned with a task-specific head (e.g. classification). MAE pre-training + fine-tuning consistently outperforms training from scratch, especially in low-data regimes.",
  },
];

const phaseColors = {
  INPUT: "#f87171",
  ENCODER: "#22d3ee",
  DECODER: "#a78bfa",
  OUTPUT: "#fb7185",
};

function PatchGrid({ showMask, highlight, size = 196 }) {
  const cols = 14;
  return (
    <div style={{
      display: "grid",
      gridTemplateColumns: `repeat(${cols}, 1fr)`,
      gap: 2,
      width: "100%",
      maxWidth: 320,
    }}>
      {Array.from({ length: size }).map((_, i) => {
        const masked = isMasked(i);
        const visible = !masked;
        let bg, border;
        if (showMask) {
          bg = masked ? "#0a0a12" : PATCH_COLORS[i % PATCH_COLORS.length] + "66";
          border = masked ? "1px solid #1a1a2a" : `1px solid ${PATCH_COLORS[i % PATCH_COLORS.length]}aa`;
        } else {
          bg = PATCH_COLORS[i % PATCH_COLORS.length] + "44";
          border = `1px solid ${PATCH_COLORS[i % PATCH_COLORS.length]}77`;
        }
        return (
          <div key={i} style={{
            aspectRatio: "1",
            background: bg,
            border,
            borderRadius: 2,
            transition: "all 0.4s ease",
          }} />
        );
      })}
    </div>
  );
}

function TokenBar({ count, color, label, dim }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap", justifyContent: "center", maxWidth: 320 }}>
        {Array.from({ length: Math.min(count, 60) }).map((_, i) => (
          <div key={i} style={{
            width: 8, height: 28,
            background: `${color}${Math.round(40 + (i / count) * 160).toString(16).padStart(2, "0")}`,
            borderRadius: 2,
            transition: "all 0.3s",
          }} />
        ))}
        {count > 60 && <span style={{ fontSize: 9, color: "#5a5a7a", alignSelf: "center" }}>+{count - 60}</span>}
      </div>
      <div style={{ fontSize: 10, color: "#6a6a8a" }}>{count} tokens × {dim} dim</div>
    </div>
  );
}

function ShapeTag({ shape, color }) {
  return (
    <div style={{
      display: "inline-flex", alignItems: "center", gap: 6,
      background: `${color}0a`, border: `1px solid ${color}44`,
      borderRadius: 8, padding: "6px 14px",
      fontFamily: "'Courier New', monospace", fontSize: 12,
      color: color, boxShadow: `0 0 16px ${color}18`,
      backdropFilter: "blur(8px)",
    }}>
      <span style={{ color: "#4a4a6a", fontSize: 10 }}>shape:</span> {shape}
    </div>
  );
}

function StepContent({ step }) {
  switch (step.id) {
    case "rgb":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 20 }}>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
            {[
              { label: "R", grad: ["#2d0000", "#8b0000", "#ff4444", "#ff8888"] },
              { label: "G", grad: ["#002d00", "#006600", "#44ff44", "#88ff88"] },
              { label: "B", grad: ["#00002d", "#000088", "#4444ff", "#8888ff"] },
            ].map((ch, ci) => (
              <div key={ci} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
                <div style={{
                  width: 100, height: 100,
                  background: `linear-gradient(135deg, ${ch.grad[0]}, ${ch.grad[1]} 40%, ${ch.grad[2]} 75%, ${ch.grad[3]})`,
                  borderRadius: 10, border: `2px solid ${ch.grad[2]}55`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  boxShadow: `0 4px 24px ${ch.grad[2]}33, inset 0 1px 0 ${ch.grad[3]}22`,
                  fontSize: 30, fontWeight: 900, color: ch.grad[3], opacity: 0.95,
                }}>
                  {ch.label}
                </div>
                <span style={{ fontSize: 10, color: "#5a5a7a", letterSpacing: 2 }}>224×224</span>
              </div>
            ))}
          </div>
          <div style={{ fontSize: 12, color: "#5a5a7a", textAlign: "center" }}>
            3 channels stacked → <span style={{ color: "#f87171" }}>150,528 raw values</span> per image
          </div>
        </div>
      );

    case "patches":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <PatchGrid showMask={false} />
          <div style={{ fontSize: 12, color: "#5a5a7a", textAlign: "center" }}>
            14 × 14 = <span style={{ color: "#fb923c" }}>196 patches</span>, each covering 16×16 px
          </div>
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", justifyContent: "center" }}>
            {[
              { top: "kernel = 16", bot: "patch size" },
              { top: "stride = 16", bot: "no overlap" },
              { top: "196 positions", bot: "14×14 grid" },
            ].map((item, idx) => (
              <div key={idx} style={{
                background: "linear-gradient(135deg, #fb923c08, #fb923c15)",
                border: "1px solid #fb923c33", borderRadius: 8,
                padding: "10px 16px", fontSize: 11, color: "#fb923c", textAlign: "center",
                boxShadow: "0 2px 12px #fb923c0a",
              }}>
                {item.top}<br /><span style={{ color: "#5a5a7a", fontSize: 10 }}>{item.bot}</span>
              </div>
            ))}
          </div>
        </div>
      );

    case "embed":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 1 }}>RAW PATCH</div>
              <div style={{ display: "flex", gap: 2 }}>
                {["#f87171", "#4ade80", "#60a5fa"].map((c, i) => (
                  <div key={i} style={{ width: 30, height: 56, background: c + "22", border: `1px solid ${c}55`, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: c }}>
                    256
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 9, color: "#5a5a7a" }}>768 values</div>
            </div>
            <div style={{ fontSize: 22, color: "#4a4a6a" }}>⊗</div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 1 }}>768 FILTERS</div>
              {[0, 1, 2, "…", 767].map((f, fi) => (
                f === "…" ? <div key={fi} style={{ fontSize: 14, color: "#4a4a6a", lineHeight: 1 }}>⋮</div> :
                  <div key={fi} style={{ width: 120, height: 16, background: "linear-gradient(90deg, #facc1511, #facc1522)", border: "1px solid #facc1544", borderRadius: 3, fontSize: 8, color: "#facc15", display: "flex", alignItems: "center", paddingLeft: 6 }}>
                    filter_{Number(f) + 1} → 1 value
                  </div>
              ))}
            </div>
            <div style={{ fontSize: 22, color: "#4a4a6a" }}>→</div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 2 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 1 }}>EMBEDDING</div>
              <div style={{ display: "flex", gap: 1 }}>
                {Array.from({ length: 32 }).map((_, i) => (
                  <div key={i} style={{ width: 5, height: 56, background: `hsl(${250 + i * 3.5}, 75%, ${35 + Math.sin(i) * 15}%)`, borderRadius: 1 }} />
                ))}
              </div>
              <div style={{ fontSize: 9, color: "#5a5a7a" }}>768-dim vector</div>
            </div>
          </div>
          <TokenBar count={49} color="#facc15" label="tokens" dim={768} />
        </div>
      );

    case "mask":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <PatchGrid showMask={true} />
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <div style={{ width: 14, height: 14, background: "#4ade8055", border: "1px solid #4ade8099", borderRadius: 3 }} />
              <span style={{ color: "#4ade80" }}>~49 visible (25%)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}>
              <div style={{ width: 14, height: 14, background: "#0a0a12", border: "1px solid #1a1a2a", borderRadius: 3 }} />
              <span style={{ color: "#4a4a6a" }}>~147 masked (75%)</span>
            </div>
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", textAlign: "center", maxWidth: 300 }}>
            Only visible tokens enter the encoder — <span style={{ color: "#4ade80" }}>4× fewer tokens</span> = massive speedup
          </div>
        </div>
      );

    case "posenc":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 4, maxWidth: 280 }}>
            {Array.from({ length: 49 }).map((_, i) => {
              const row = Math.floor(i / 7), col = i % 7;
              const hue = (row * 30 + col * 15) % 360;
              return (
                <div key={i} style={{
                  aspectRatio: "1", borderRadius: 4,
                  background: `linear-gradient(135deg, hsl(${hue}, 60%, 20%), hsl(${hue}, 65%, 30%))`,
                  border: `1px solid hsl(${hue}, 60%, 45%)`,
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 8, color: `hsl(${hue}, 80%, 72%)`,
                  boxShadow: `0 2px 8px hsla(${hue}, 60%, 30%, 0.3)`,
                }}>
                  {row},{col}
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", textAlign: "center" }}>
            Each token gets a <span style={{ color: "#34d399" }}>(row, col)</span> position encoding added
          </div>
          <div style={{
            background: "linear-gradient(135deg, #34d39908, #34d39915)",
            border: "1px solid #34d39933", borderRadius: 8,
            padding: "10px 16px", fontSize: 11, color: "#34d399", textAlign: "center",
            boxShadow: "0 2px 12px #34d3990a",
          }}>
            sin/cos(position × freq) added element-wise<br />
            <span style={{ color: "#5a5a7a" }}>shape unchanged — just enriched with location</span>
          </div>
        </div>
      );

    case "encoder":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 14 }}>
          {[
            { label: "Block 1", sub: "LN → MHSA → + → LN → FFN → +", col: "#22d3ee" },
            { label: "Block 2", sub: "LN → MHSA → + → LN → FFN → +", col: "#22d3ee" },
            { label: "⋮", sub: "", col: "#4a4a6a" },
            { label: "Block 12", sub: "LN → MHSA → + → LN → FFN → +", col: "#22d3ee" },
          ].map((b, i) => (
            b.label === "⋮" ?
              <div key={i} style={{ fontSize: 20, color: "#4a4a6a" }}>⋮</div> :
              <div key={i} style={{
                width: "100%", maxWidth: 380,
                background: `linear-gradient(135deg, ${b.col}08, ${b.col}12)`,
                border: `1px solid ${b.col}33`,
                borderRadius: 10, padding: "12px 18px",
                display: "flex", justifyContent: "space-between", alignItems: "center",
                boxShadow: `0 2px 12px ${b.col}0a`,
              }}>
                <span style={{ fontSize: 12, color: b.col, fontWeight: 700 }}>{b.label}</span>
                <span style={{ fontSize: 9, color: "#5a5a7a", fontFamily: "monospace" }}>{b.sub}</span>
              </div>
          ))}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center", marginTop: 4 }}>
            {[
              { top: "12 blocks", bot: "ViT-Base" },
              { top: "12 heads", bot: "dim_head=64" },
              { top: "dim = 768", bot: "MLP: 768→3072→768" },
              { top: "GELU", bot: "activation" },
            ].map((item, idx) => (
              <div key={idx} style={{
                background: "linear-gradient(135deg, #22d3ee08, #22d3ee15)",
                border: "1px solid #22d3ee33", borderRadius: 8,
                padding: "8px 12px", fontSize: 10, color: "#22d3ee", textAlign: "center",
              }}>
                {item.top}<br /><span style={{ color: "#5a5a7a", fontSize: 9 }}>{item.bot}</span>
              </div>
            ))}
          </div>
        </div>
      );

    case "enc_projection":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 1 }}>ENCODER OUT</div>
              <div style={{ display: "flex", gap: 1 }}>
                {Array.from({ length: 24 }).map((_, i) => (
                  <div key={i} style={{ width: 5, height: 52, background: `hsl(${185 + i * 2}, 70%, ${30 + Math.sin(i) * 12}%)`, borderRadius: 1 }} />
                ))}
              </div>
              <div style={{ fontSize: 9, color: "#5a5a7a" }}>768-dim</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{ fontSize: 18, color: "#6366f1" }}>→</div>
              <div style={{
                background: "linear-gradient(135deg, #6366f108, #6366f118)",
                border: "1px solid #6366f133", borderRadius: 8,
                padding: "8px 14px", fontSize: 10, color: "#6366f1", textAlign: "center",
              }}>
                Linear<br /><span style={{ color: "#5a5a7a", fontSize: 9 }}>W: 768×512</span>
              </div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 1 }}>DECODER IN</div>
              <div style={{ display: "flex", gap: 1 }}>
                {Array.from({ length: 16 }).map((_, i) => (
                  <div key={i} style={{ width: 5, height: 52, background: `hsl(${245 + i * 3}, 70%, ${30 + Math.sin(i) * 12}%)`, borderRadius: 1 }} />
                ))}
              </div>
              <div style={{ fontSize: 9, color: "#5a5a7a" }}>512-dim</div>
            </div>
          </div>
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ background: "linear-gradient(135deg, #6366f108, #6366f115)", border: "1px solid #6366f133", borderRadius: 8, padding: "8px 14px", fontSize: 10, color: "#6366f1", textAlign: "center" }}>
              393K params<br /><span style={{ color: "#5a5a7a", fontSize: 9 }}>768 × 512 + bias</span>
            </div>
            <div style={{ background: "linear-gradient(135deg, #6366f108, #6366f115)", border: "1px solid #6366f133", borderRadius: 8, padding: "8px 14px", fontSize: 10, color: "#6366f1", textAlign: "center" }}>
              ~49 tokens<br /><span style={{ color: "#5a5a7a", fontSize: 9 }}>visible only</span>
            </div>
          </div>
        </div>
      );

    case "decode_tokens":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(14, 1fr)", gap: 2, width: "100%", maxWidth: 320 }}>
            {Array.from({ length: 196 }).map((_, i) => {
              const masked = isMasked(i);
              return (
                <div key={i} style={{
                  aspectRatio: "1", borderRadius: 2,
                  background: masked ? "#818cf833" : PATCH_COLORS[i % PATCH_COLORS.length] + "66",
                  border: masked ? "1px solid #818cf855" : `1px solid ${PATCH_COLORS[i % PATCH_COLORS.length]}99`,
                }} />
              );
            })}
          </div>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
              <div style={{ width: 12, height: 12, background: "#4ade8055", border: "1px solid #4ade8099", borderRadius: 3 }} />
              <span style={{ color: "#4ade80" }}>encoder output</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
              <div style={{ width: 12, height: 12, background: "#818cf833", border: "1px solid #818cf855", borderRadius: 3 }} />
              <span style={{ color: "#818cf8" }}>[MASK] tokens</span>
            </div>
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", textAlign: "center" }}>
            Full 196-token sequence restored — projected to <span style={{ color: "#818cf8" }}>dim=512</span>
          </div>
        </div>
      );

    case "decode_pos":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(14, 1fr)", gap: 2, width: "100%", maxWidth: 320 }}>
            {Array.from({ length: 196 }).map((_, i) => {
              const row = Math.floor(i / 14), col = i % 14;
              const hue = (row * 18 + col * 9) % 360;
              return (
                <div key={i} style={{
                  aspectRatio: "1", borderRadius: 2,
                  background: `linear-gradient(135deg, hsla(${hue}, 55%, 25%, 0.6), hsla(${hue}, 60%, 35%, 0.7))`,
                  border: `1px solid hsla(${hue}, 55%, 50%, 0.5)`,
                }} />
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", textAlign: "center" }}>
            All 196 tokens — <span style={{ color: "#a78bfa" }}>including mask tokens</span> — get positional encodings.<br />
            This is the only spatial signal mask tokens receive.
          </div>
        </div>
      );

    case "decoder":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 12 }}>
          {[
            { label: "Block 1", sub: "LN → MHSA → + → LN → FFN → +", col: "#c084fc" },
            { label: "⋮", sub: "", col: "#4a4a6a" },
            { label: "Block 8", sub: "LN → MHSA → + → LN → FFN → +", col: "#c084fc" },
          ].map((b, i) => (
            b.label === "⋮" ?
              <div key={i} style={{ fontSize: 20, color: "#4a4a6a" }}>⋮</div> :
              <div key={i} style={{
                width: "100%", maxWidth: 380,
                background: `linear-gradient(135deg, ${b.col}08, ${b.col}12)`,
                border: `1px solid ${b.col}33`,
                borderRadius: 10, padding: "12px 18px",
                display: "flex", justifyContent: "space-between", alignItems: "center",
                boxShadow: `0 2px 12px ${b.col}0a`,
              }}>
                <span style={{ fontSize: 12, color: b.col, fontWeight: 700 }}>{b.label}</span>
                <span style={{ fontSize: 9, color: "#5a5a7a", fontFamily: "monospace" }}>{b.sub}</span>
              </div>
          ))}
          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", justifyContent: "center", marginTop: 6 }}>
            <div style={{ background: "linear-gradient(135deg, #c084fc08, #c084fc15)", border: "1px solid #c084fc33", borderRadius: 8, padding: "8px 14px", fontSize: 10, color: "#c084fc", textAlign: "center" }}>
              8 blocks<br /><span style={{ color: "#5a5a7a" }}>vs encoder's 12</span>
            </div>
            <div style={{ background: "linear-gradient(135deg, #c084fc08, #c084fc15)", border: "1px solid #c084fc33", borderRadius: 8, padding: "8px 14px", fontSize: 10, color: "#c084fc", textAlign: "center" }}>
              dim = 512<br /><span style={{ color: "#5a5a7a" }}>vs encoder's 768</span>
            </div>
            <div style={{ background: "linear-gradient(135deg, #c084fc08, #c084fc15)", border: "1px solid #c084fc33", borderRadius: 8, padding: "8px 14px", fontSize: 10, color: "#c084fc", textAlign: "center" }}>
              Self-Attention<br /><span style={{ color: "#5a5a7a" }}>not cross-attn</span>
            </div>
          </div>
        </div>
      );

    case "reconstruct":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div style={{ width: "100%", maxWidth: 320 }}>
            <div style={{
              background: "linear-gradient(135deg, #f472b608, #f472b615)",
              border: "1px solid #f472b633", borderRadius: 10,
              padding: "12px 16px", fontSize: 11, color: "#f472b6",
              marginBottom: 12, textAlign: "center",
              boxShadow: "0 2px 12px #f472b60a",
            }}>
              Linear: 512 → 768 (= 16×16×3 pixels per patch)
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(14, 1fr)", gap: 2 }}>
              {Array.from({ length: 196 }).map((_, i) => {
                const masked = isMasked(i);
                return (
                  <div key={i} style={{
                    aspectRatio: "1", borderRadius: 2,
                    background: masked ? "#f472b644" : PATCH_COLORS[i % PATCH_COLORS.length] + "66",
                    border: masked ? "1px solid #f472b677" : `1px solid ${PATCH_COLORS[i % PATCH_COLORS.length]}99`,
                  }} />
                );
              })}
            </div>
          </div>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
              <div style={{ width: 12, height: 12, background: "#f472b644", border: "1px solid #f472b677", borderRadius: 3 }} />
              <span style={{ color: "#f472b6" }}>predicted pixels (loss here)</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 11 }}>
              <div style={{ width: 12, height: 12, background: "#4ade8055", border: "1px solid #4ade8099", borderRadius: 3 }} />
              <span style={{ color: "#4ade80" }}>visible (no loss)</span>
            </div>
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", textAlign: "center" }}>
            MSE loss computed <span style={{ color: "#f472b6" }}>only on masked patches</span>
          </div>
        </div>
      );

    case "output":
      return (
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 18 }}>
          <div style={{ display: "flex", gap: 20, flexWrap: "wrap", justifyContent: "center", alignItems: "center" }}>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 2 }}>MASKED INPUT</div>
              <div style={{
                width: 120, height: 120,
                background: "linear-gradient(135deg, #0a0a14, #0d0d1a)",
                border: "1px solid #1a1a2a", borderRadius: 8,
                display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 1.5, padding: 5,
                boxShadow: "0 4px 20px rgba(0,0,0,0.5)",
              }}>
                {Array.from({ length: 49 }).map((_, i) => (
                  <div key={i} style={{ background: isMasked(i * 4) ? "#0a0a12" : PATCH_COLORS[i % PATCH_COLORS.length] + "88", borderRadius: 1.5 }} />
                ))}
              </div>
            </div>
            <div style={{ display: "flex", alignItems: "center", fontSize: 24, color: "#4a4a6a" }}>→</div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
              <div style={{ fontSize: 10, color: "#6a6a8a", letterSpacing: 2 }}>RECONSTRUCTED</div>
              <div style={{
                width: 120, height: 120,
                background: "linear-gradient(135deg, #0a0a14, #0d0d1a)",
                border: "1px solid #fb7185", borderRadius: 8,
                boxShadow: "0 0 30px #fb718544, 0 4px 20px rgba(0,0,0,0.5)",
                display: "grid", gridTemplateColumns: "repeat(7,1fr)", gap: 1.5, padding: 5,
              }}>
                {Array.from({ length: 49 }).map((_, i) => (
                  <div key={i} style={{ background: PATCH_COLORS[i % PATCH_COLORS.length] + "aa", borderRadius: 1.5 }} />
                ))}
              </div>
            </div>
          </div>
          <div style={{
            background: "linear-gradient(135deg, #fb718508, #fb718515)",
            border: "1px solid #fb718533", borderRadius: 10,
            padding: "12px 18px", fontSize: 11, color: "#fb7185",
            textAlign: "center", maxWidth: 300,
            boxShadow: "0 2px 12px #fb71850a",
          }}>
            At inference: <span style={{ color: "#e8e6f0" }}>decoder is discarded</span>.<br />
            Encoder alone is fine-tuned on downstream tasks.
          </div>
        </div>
      );

    default:
      return null;
  }
}

export default function MAEVisualizer() {
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  const timerRef = useRef(null);
  const total = STEPS.length;

  useEffect(() => {
    if (autoPlay) {
      timerRef.current = setInterval(() => {
        setStep(s => {
          if (s >= total - 1) { setAutoPlay(false); return s; }
          return s + 1;
        });
      }, 3000);
    }
    return () => clearInterval(timerRef.current);
  }, [autoPlay]);

  const cur = STEPS[step];

  const phaseGroups = {
    INPUT: STEPS.filter(s => s.phase === "INPUT"),
    ENCODER: STEPS.filter(s => s.phase === "ENCODER"),
    DECODER: STEPS.filter(s => s.phase === "DECODER"),
    OUTPUT: STEPS.filter(s => s.phase === "OUTPUT"),
  };

  return (
    <div style={{
      minHeight: "100vh",
      width: "100%",
      background: "radial-gradient(ellipse at 50% 0%, #0f0f24 0%, #0a0a16 35%, #07070f 70%)",
      color: "#e0dff0",
      fontFamily: "'Courier New', monospace",
      boxSizing: "border-box",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
    }}>
      {/* Header */}
      <div style={{
        width: "100%",
        maxWidth: 1100,
        padding: "40px 24px 0",
        textAlign: "center",
        boxSizing: "border-box",
      }}>
        <div style={{ fontSize: 10, letterSpacing: 8, color: "#4a4a6a", marginBottom: 10, textTransform: "uppercase" }}>
          Interactive Explainer
        </div>
        <h1 style={{
          margin: 0, fontSize: "clamp(22px, 5vw, 38px)", fontWeight: 900, letterSpacing: -1,
          background: "linear-gradient(135deg, #ff6b6b 0%, #ffa94d 15%, #ffd43b 30%, #4ade80 45%, #22d3ee 60%, #818cf8 75%, #c084fc 85%, #f472b6 100%)",
          WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent",
          filter: "drop-shadow(0 0 20px rgba(168, 139, 250, 0.3))",
        }}>
          Masked Autoencoder
        </h1>
        <div style={{ fontSize: 11, color: "#4a4a6a", marginTop: 8, letterSpacing: 4 }}>
          FULL ARCHITECTURE · RGB → ENCODER → DECODER → PIXELS
        </div>
      </div>

      {/* Phase nav */}
      <div style={{
        width: "100%", maxWidth: 1100,
        padding: "24px 24px 0",
        boxSizing: "border-box",
        display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "center",
      }}>
        {Object.entries(phaseGroups).map(([phase, steps]) => (
          <div key={phase} style={{ display: "flex", alignItems: "center", gap: 5 }}>
            <div style={{
              fontSize: 9, letterSpacing: 2, color: phaseColors[phase],
              background: `linear-gradient(135deg, ${phaseColors[phase]}10, ${phaseColors[phase]}1a)`,
              border: `1px solid ${phaseColors[phase]}33`,
              padding: "4px 10px", borderRadius: 20,
              boxShadow: `0 0 8px ${phaseColors[phase]}0a`,
            }}>
              {phase}
            </div>
            <div style={{ display: "flex", gap: 4 }}>
              {steps.map(s => {
                const idx = STEPS.indexOf(s);
                const active = idx === step;
                const passed = idx < step;
                return (
                  <button key={s.id} onClick={() => { setAutoPlay(false); setStep(idx); }} style={{
                    width: 28, height: 28, borderRadius: 6, border: "none",
                    background: active
                      ? `linear-gradient(135deg, ${s.color}cc, ${s.color}88)`
                      : passed ? s.color + "33" : "#1a1a2a",
                    cursor: "pointer", transition: "all 0.25s ease",
                    outline: active ? `2px solid ${s.color}` : "none",
                    outlineOffset: 2,
                    boxShadow: active ? `0 0 12px ${s.color}44` : "none",
                  }} title={s.label} />
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Main card */}
      <div style={{
        width: "100%", maxWidth: 1100,
        padding: "24px",
        boxSizing: "border-box",
        display: "flex", flexDirection: "column", gap: 20,
      }}>
        <div style={{
          background: "linear-gradient(135deg, rgba(12, 12, 28, 0.9), rgba(14, 14, 32, 0.95))",
          border: `1px solid ${cur.color}33`,
          borderRadius: 16,
          boxShadow: `0 0 60px ${cur.color}15, 0 4px 30px rgba(0, 0, 0, 0.4)`,
          overflow: "hidden",
          backdropFilter: "blur(12px)",
          transition: "border-color 0.4s ease, box-shadow 0.4s ease",
        }}>
          {/* Card header */}
          <div style={{
            borderBottom: `1px solid ${cur.color}22`,
            padding: "16px 24px",
            display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 10,
            background: `linear-gradient(90deg, ${cur.color}08, transparent 50%)`,
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div style={{
                width: 32, height: 32, borderRadius: 8,
                background: `linear-gradient(135deg, ${cur.color}22, ${cur.color}33)`,
                border: `1px solid ${cur.color}55`,
                display: "flex", alignItems: "center", justifyContent: "center",
                fontSize: 12, color: cur.color, fontWeight: 700,
                boxShadow: `0 0 10px ${cur.color}22`,
              }}>
                {String(step + 1).padStart(2, "0")}
              </div>
              <div>
                <div style={{ fontSize: 14, fontWeight: 700, color: cur.color }}>{cur.label}</div>
                <div style={{ fontSize: 9, color: "#4a4a6a", letterSpacing: 3 }}>{cur.phase}</div>
              </div>
            </div>
            <ShapeTag shape={cur.shape} color={cur.color} />
          </div>

          {/* Card body: two columns on wide, stacked on narrow */}
          <div style={{
            display: "flex",
            flexDirection: "row",
            flexWrap: "wrap",
            gap: 0,
          }}>
            {/* Visual */}
            <div style={{
              flex: "1 1 340px",
              padding: "28px 24px",
              display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
              borderRight: `1px solid ${cur.color}11`,
              minWidth: 0,
              animation: "fadeSlideIn 0.4s ease-out",
            }}>
              <StepContent step={cur} />
            </div>

            {/* Text panel */}
            <div style={{
              flex: "1 1 300px",
              padding: "28px 24px",
              display: "flex", flexDirection: "column", justifyContent: "space-between", gap: 20,
              minWidth: 0,
              animation: "fadeSlideIn 0.4s ease-out 0.1s both",
            }}>
              <div>
                <div style={{ fontSize: 9, letterSpacing: 3, color: "#4a4a6a", marginBottom: 10 }}>WHAT'S HAPPENING</div>
                <p style={{ fontSize: 13, color: "#a09eb8", lineHeight: 1.85, margin: 0 }}>{cur.desc}</p>
              </div>
              <div style={{
                background: `linear-gradient(135deg, ${cur.color}08, ${cur.color}12)`,
                border: `1px solid ${cur.color}22`,
                borderRadius: 10, padding: "14px 18px",
                boxShadow: `0 2px 12px ${cur.color}08`,
              }}>
                <div style={{ fontSize: 9, letterSpacing: 2, color: cur.color, marginBottom: 6 }}>💡 KEY INSIGHT</div>
                <p style={{ fontSize: 12, color: "#7a7a9a", margin: 0, lineHeight: 1.75 }}>{cur.tip}</p>
              </div>
            </div>
          </div>
        </div>

        {/* Progress bar */}
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{ flex: 1, height: 5, background: "#1a1a2a", borderRadius: 3, overflow: "hidden", boxShadow: "inset 0 1px 3px rgba(0,0,0,0.4)" }}>
            <div style={{
              height: "100%", borderRadius: 3,
              background: "linear-gradient(90deg, #ff6b6b, #ffa94d, #ffd43b, #4ade80, #22d3ee, #818cf8, #c084fc, #f472b6)",
              width: `${((step + 1) / total) * 100}%`,
              transition: "width 0.4s ease",
              boxShadow: "0 0 12px rgba(168, 139, 250, 0.3), 0 0 4px rgba(168, 139, 250, 0.2)",
            }} />
          </div>
          <div style={{ fontSize: 11, color: "#5a5a7a", whiteSpace: "nowrap", fontWeight: 600 }}>{step + 1} / {total}</div>
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 10, justifyContent: "center", flexWrap: "wrap" }}>
          <button onClick={() => { setAutoPlay(false); setStep(0); }} style={{
            padding: "9px 18px", borderRadius: 8, fontFamily: "inherit", fontSize: 11,
            border: "1px solid #2a2a4a", background: "linear-gradient(135deg, #0d0d1a, #12122a)",
            color: "#5a5a7a", cursor: "pointer", letterSpacing: 2,
            transition: "all 0.25s ease",
          }}>⏮ RESET</button>
          <button onClick={() => { setAutoPlay(false); setStep(s => Math.max(0, s - 1)); }} disabled={step === 0} style={{
            padding: "9px 20px", borderRadius: 8, fontFamily: "inherit", fontSize: 11,
            border: "1px solid #2a2a4a", background: "linear-gradient(135deg, #0d0d1a, #12122a)",
            color: step === 0 ? "#2a2a3a" : "#e0dff0", cursor: step === 0 ? "not-allowed" : "pointer",
            letterSpacing: 2, transition: "all 0.25s ease",
          }}>← PREV</button>
          <button onClick={() => setAutoPlay(a => !a)} style={{
            padding: "9px 24px", borderRadius: 8, fontFamily: "inherit", fontSize: 11,
            border: `1px solid ${autoPlay ? "#f87171" : cur.color}55`,
            background: autoPlay
              ? "linear-gradient(135deg, #f8717115, #f8717122)"
              : `linear-gradient(135deg, ${cur.color}12, ${cur.color}1a)`,
            color: autoPlay ? "#f87171" : cur.color,
            cursor: "pointer", letterSpacing: 2, minWidth: 90,
            boxShadow: `0 0 12px ${autoPlay ? "#f87171" : cur.color}11`,
            transition: "all 0.25s ease",
          }}>
            {autoPlay ? "⏸ PAUSE" : "▶ AUTO"}
          </button>
          <button onClick={() => { setAutoPlay(false); setStep(s => Math.min(total - 1, s + 1)); }} disabled={step === total - 1} style={{
            padding: "9px 20px", borderRadius: 8, fontFamily: "inherit", fontSize: 11,
            border: "1px solid #2a2a4a", background: "linear-gradient(135deg, #0d0d1a, #12122a)",
            color: step === total - 1 ? "#2a2a3a" : "#e0dff0", cursor: step === total - 1 ? "not-allowed" : "pointer",
            letterSpacing: 2, transition: "all 0.25s ease",
          }}>NEXT →</button>
        </div>

        {/* Step dot nav */}
        <div style={{ display: "flex", gap: 5, justifyContent: "center", flexWrap: "wrap" }}>
          {STEPS.map((s, i) => (
            <button key={s.id} onClick={() => { setAutoPlay(false); setStep(i); }} title={s.label} style={{
              width: i === step ? 28 : 10, height: 10, borderRadius: 5,
              background: i === step
                ? `linear-gradient(90deg, ${s.color}cc, ${s.color}88)`
                : i < step ? s.color + "55" : "#1a1a2a",
              border: "none", cursor: "pointer", transition: "all 0.3s ease", padding: 0,
              boxShadow: i === step ? `0 0 8px ${s.color}44` : "none",
            }} />
          ))}
        </div>

        {/* Architecture summary ribbon */}
        <div style={{
          background: "linear-gradient(135deg, rgba(12, 12, 28, 0.8), rgba(14, 14, 32, 0.9))",
          border: "1px solid #1a1a2a", borderRadius: 14,
          padding: "16px 20px", display: "flex", flexWrap: "wrap", gap: 5, alignItems: "center", justifyContent: "center",
          backdropFilter: "blur(8px)",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
        }}>
          {STEPS.map((s, i) => (
            <div key={s.id} style={{ display: "flex", alignItems: "center", gap: 5 }}>
              <button onClick={() => { setAutoPlay(false); setStep(i); }} style={{
                padding: "5px 12px", borderRadius: 20, fontFamily: "inherit", fontSize: 10,
                border: `1px solid ${step === i ? s.color : s.color + "33"}`,
                background: step === i
                  ? `linear-gradient(135deg, ${s.color}22, ${s.color}33)`
                  : "transparent",
                color: step === i ? s.color : "#4a4a6a",
                cursor: "pointer", transition: "all 0.25s", letterSpacing: 1, whiteSpace: "nowrap",
                boxShadow: step === i ? `0 0 8px ${s.color}22` : "none",
              }}>
                {s.label}
              </button>
              {i < STEPS.length - 1 && <span style={{ color: "#2a2a4a", fontSize: 11 }}>›</span>}
            </div>
          ))}
        </div>
      </div>

      <div style={{ height: 40 }} />
    </div>
  );
}