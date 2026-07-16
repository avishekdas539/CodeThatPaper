# 🚀 TurboQuant MSE Benchmark Report

> **Configuration**

| Parameter | Value |
|-----------|-------|
| **Vectors/Test** | **4,096** |
| **Dimensions** | **1024, 1536, 2048** |
| **Bit Widths** | **1, 2, 3, 4** |
| **Distribution** | Gaussian |
| **σ (Std)** | 10 |
| **Seed** | 42 |

---

## 🧮 Centroid Optimization

| Bits | Status |
|------|--------|
| **1** | 🟢 Converged in **1** iteration |
| **2** | 🟢 Converged in **32** iterations |
| **3** | 🟡 Maximum iterations reached |
| **4** | 🟡 Maximum iterations reached |

---

# 📊 Benchmark Results

## Dimension = 1024

| Bits | MSE | Compression | Encode (s) | Decode (s) | Encode (vec/s) | Decode (vec/s) |
|----:|----:|-----------:|-----------:|-----------:|---------------:|---------------:|
| <span style="color:#2ecc71"><b>1</b></span> | <span style="color:#e74c3c"><b>0.363145</b></span> | <span style="color:#27ae60"><b>32×</b></span> | <span style="color:#27ae60"><b>11.04</b></span> | <span style="color:#27ae60"><b>1.31</b></span> | <span style="color:#27ae60"><b>371</b></span> | <span style="color:#27ae60"><b>3135</b></span> |
| 2 | 0.117226 | 16× | 11.30 | 1.56 | 363 | 2623 |
| 3 | 0.034400 | 8× | 11.73 | 2.01 | 349 | 2040 |
| <span style="color:#3498db"><b>4</b></span> | <span style="color:#27ae60"><b>0.009502</b></span> | 8× | 11.74 | 1.99 | 349 | 2053 |

---

## Dimension = 1536

| Bits | MSE | Compression | Encode (s) | Decode (s) | Encode (vec/s) | Decode (vec/s) |
|----:|----:|-----------:|-----------:|-----------:|---------------:|---------------:|
| 1 | 0.363224 | <span style="color:#27ae60"><b>32×</b></span> | <span style="color:#27ae60"><b>16.04</b></span> | <span style="color:#27ae60"><b>2.04</b></span> | <span style="color:#27ae60"><b>255</b></span> | <span style="color:#27ae60"><b>2006</b></span> |
| 2 | 0.117458 | 16× | 17.06 | 2.39 | 240 | 1716 |
| 3 | 0.034522 | 8× | 18.05 | 3.12 | 227 | 1312 |
| <span style="color:#3498db"><b>4</b></span> | <span style="color:#27ae60"><b>0.009533</b></span> | 8× | 17.88 | 3.02 | 229 | 1358 |

---

## Dimension = 2048

| Bits | MSE | Compression | Encode (s) | Decode (s) | Encode (vec/s) | Decode (vec/s) |
|----:|----:|-----------:|-----------:|-----------:|---------------:|---------------:|
| 1 | 0.363212 | <span style="color:#27ae60"><b>32×</b></span> | <span style="color:#27ae60"><b>21.72</b></span> | <span style="color:#27ae60"><b>2.87</b></span> | <span style="color:#27ae60"><b>189</b></span> | <span style="color:#27ae60"><b>1427</b></span> |
| 2 | 0.117399 | 16× | 23.29 | 3.27 | 176 | 1254 |
| 3 | 0.034522 | 8× | 24.14 | 4.15 | 170 | 988 |
| <span style="color:#3498db"><b>4</b></span> | <span style="color:#27ae60"><b>0.009540</b></span> | 8× | 24.71 | 4.09 | 166 | 1001 |

---

# 📈 MSE Overview

| Bits | 1024 | 1536 | 2048 |
|------|------|------|------|
| 1 | 🔴 **0.363145** | 🔴 **0.363224** | 🔴 **0.363212** |
| 2 | 🟠 0.117226 | 🟠 0.117458 | 🟠 0.117399 |
| 3 | 🟡 0.034400 | 🟡 0.034522 | 🟡 0.034522 |
| 4 | 🟢 **0.009502** | 🟢 **0.009533** | 🟢 **0.009540** |

---

# 🏆 Best Configurations

| Metric | Configuration | Value |
|--------|---------------|------:|
| 🟢 Lowest MSE | **4-bit @ 1024** | **0.009502** |
| 🟢 Best Compression | **1-bit @ 1024** | **32×** |
| ⚡ Fastest Encode | **1-bit @ 1024** | **371 vec/s** |
| ⚡ Fastest Decode | **1-bit @ 1024** | **3135 vec/s** |

---

# 📌 Summary

✅ **Best reconstruction quality:** **4-bit quantization**

✅ **Highest compression:** **1-bit (32×)**

✅ **Fastest encoding:** **371 vectors/s**

✅ **Fastest decoding:** **3135 vectors/s**

---

### Color Legend

| Color | Meaning |
|-------|---------|
| 🟢 Green | Best |
| 🟡 Yellow | Medium |
| 🔴 Red | Worst |
| ⚡ Blue | Performance Highlight |