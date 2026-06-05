import os
import numpy as np
import matplotlib.pyplot as plt


RESULT_DIR = "./lab10/results"
os.makedirs(RESULT_DIR, exist_ok=True)

# ==========================
# 1. Sinusoidal Position Encoding
# ==========================
def sinusoidal_position_encoding(seq_len, d_model):

    pe = np.zeros((seq_len, d_model))

    position = np.arange(seq_len).reshape(-1, 1)

    div_term = np.exp(
        np.arange(0, d_model, 2)
        * -(np.log(10000.0) / d_model)
    )

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)

    return pe

pe = sinusoidal_position_encoding(100, 64)

plt.figure(figsize=(8,4))
plt.imshow(pe, aspect='auto')
plt.colorbar()
plt.title("Sinusoidal Position Encoding")
plt.savefig(f"{RESULT_DIR}/sinusoidal_encoding.png")
plt.close()

# ==========================
# 2. 二维向量旋转
# ==========================
v = np.array([1, 0])

theta = np.pi/4

R = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta),  np.cos(theta)]
])

v_rot = R @ v

plt.figure(figsize=(5,5))

plt.arrow(0,0,v[0],v[1],width=0.02)
plt.arrow(0,0,v_rot[0],v_rot[1],width=0.02)

plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)

plt.grid()

plt.title("2D Vector Rotation")

plt.savefig(f"{RESULT_DIR}/vector_rotation.png")
plt.close()

# ==========================
# 3. RoPE实现
# ==========================
def rope_embedding(x, position):

    d = len(x)

    x = x.copy()

    for i in range(0, d, 2):

        theta = position / (10000 ** (i / d))

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x1 = x[i]
        x2 = x[i+1]

        x[i] = x1*cos_t - x2*sin_t
        x[i+1] = x1*sin_t + x2*cos_t

    return x

x = np.random.randn(64)

rope_vectors = []

for pos in range(50):

    rope_vectors.append(
        rope_embedding(x, pos)
    )

rope_vectors = np.array(rope_vectors)

plt.figure(figsize=(8,4))
plt.imshow(rope_vectors)
plt.colorbar()
plt.title("RoPE Encoding")
plt.savefig(f"{RESULT_DIR}/rope_encoding.png")
plt.close()

# ==========================
# 4. E+Pos vs RoPE
# ==========================
token = np.random.randn(64)

epos = []
rope = []

for pos in range(50):

    pos_vec = sinusoidal_position_encoding(
        50,
        64
    )[pos]

    epos.append(token + pos_vec)

    rope.append(
        rope_embedding(token, pos)
    )

epos = np.array(epos)
rope = np.array(rope)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.imshow(epos)
plt.title("E + Pos")

plt.subplot(1,2,2)
plt.imshow(rope)
plt.title("RoPE")

plt.tight_layout()

plt.savefig(f"{RESULT_DIR}/epos_vs_rope.png")
plt.close()

# ==========================
# 5. RoPE相对位置验证
# ==========================
q = np.random.randn(64)
k = np.random.randn(64)

scores = []

for delta in range(50):

    q_rot = rope_embedding(q, 10)

    k_rot = rope_embedding(
        k,
        10 + delta
    )

    score = np.dot(q_rot, k_rot)

    scores.append(score)

plt.figure(figsize=(8,4))
plt.plot(scores)

plt.title("RoPE Relative Position Property")
plt.xlabel("Relative Distance")

plt.savefig(
    f"{RESULT_DIR}/rope_relative_property.png"
)

plt.close()

# ==========================
# 6. Attention Score对比
# ==========================
epos_scores = []
rope_scores = []

for pos in range(50):

    pos_vec = sinusoidal_position_encoding(
        50,
        64
    )[pos]

    q1 = q + pos_vec
    k1 = k + pos_vec

    epos_scores.append(
        np.dot(q1,k1)
    )

    q2 = rope_embedding(q,pos)
    k2 = rope_embedding(k,pos)

    rope_scores.append(
        np.dot(q2,k2)
    )

plt.figure(figsize=(8,4))

plt.plot(epos_scores,label='E+Pos')
plt.plot(rope_scores,label='RoPE')

plt.legend()

plt.title(
    "Attention Score Comparison"
)

plt.savefig(
    f"{RESULT_DIR}/attention_score_comparison.png"
)

plt.close()

print("Lab10实验完成")