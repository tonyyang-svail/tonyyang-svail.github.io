---
layout: post
title:  "Deep Learning System Stack"
comments: true
---

I came across this [nice summary][1] of today's deep learning system stack.

> Here is what a deep learning system stack would look like in nowdays.
> 1. Build operator level graph description language: name whatever dl frameworks you care about, and [ONNX][2]
> 2. Tensor primitive level graph description languages: [NNVM][3], [HLO/XLA][4], [NGraph][5]. It is close enough to the first one that you can also build graph optimization on first layer and bypass this layer.
> 3. DSL for description and codegen: TVM, image processing languages like [halide][6], [darkroom][7].
> 4. Hardcoded optimized kernel library: [nnpack][8], [cudnn][9], [libdnn][10]
> 5. Device dependent library: [maxas][11](assembler for NVIDIA Maxwell architecture)


To elaberate, consider convoluting an average kernel over an image, AKA blurring. And the following shows what the code looks like on each level.

Level 1 and 2: operator/tensor primitive level, we already have the `conv` operator.

```python
image = load_image()
average_kernel = np.full((3,3), 1.0/9)
blurred = conv(image, average_kernel)
```

Level 3: DSL for description and codegen. Take halide for example, a user needs to write both
1. the definition of the algorithm
2. the scheduling of storage(tile, vectorize) and computation order(parallel)

```cpp
Func halide_blur(Func in) {
  Func tmp, blurred;
  Var x, y, xi, yi;

  // The algorithm
  tmp(x, y) = (in(x-1, y) + in(x, y) + in(x+1, y))/3;
  blurred(x, y) = (tmp(x, y-1) + tmp(x, y) + tmp(x, y+1))/3;

  // The schedule
  blurred.tile(x, y, xi, yi, 256, 32)
         .vectorize(xi, 8).parallel(y);
  tmp.chunk(x).vectorize(x, 8);
  return blurred;
}
```

Level 4: Hard coded optimized kernel. A user need to hardcode vectorization, multithreading, tiling and fusion.

```cpp
void fast_blur(const Image &in, Image &blurred) {
  m128i one_third = _mm_set1_epi16(21846);
  #pragma omp parallel for
  for (int yTile = 0; yTile < in.height(); yTile += 32) {
    m128i a, b, c, sum, avg;
    m128i tmp[(256/8)*(32+2)];
    for (int xTile = 0; xTile < in.width(); xTile += 256) {
      m128i *tmpPtr = tmp;
      for (int y = -1; y < 32+1; y++) {
        const uint16_t *inPtr = &(in(xTile, yTile+y));
        for (int x = 0; x < 256; x += 8) {
          a = _mm_loadu_si128(( m128i*)(inPtr-1));
          b = _mm_loadu_si128(( m128i*)(inPtr+1));
          c = _mm_load_si128(( m128i*)(inPtr));
          sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
          avg = _mm_mulhi_epi16(sum, one_third);
          _mm_store_si128(tmpPtr++, avg);
          inPtr += 8;
        }}
      tmpPtr = tmp;
      for (int y = 0; y < 32; y++) {
        m128i *outPtr = ( m128i *)(&(blurred(xTile, yTile+y)));
        for (int x = 0; x < 256; x += 8) {
          a = _mm_load_si128(tmpPtr+(2*256)/8);
          b = _mm_load_si128(tmpPtr+256/8);
          c = _mm_load_si128(tmpPtr++);
          sum = _mm_add_epi16(_mm_add_epi16(a, b), c);
          avg = _mm_mulhi_epi16(sum, one_third);
          _mm_store_si128(outPtr++, avg);
        }
      }
    }
  }
}
```

Level 5: Device dependent library. Usual coded in assembly language, e.g. [maxas][12].

[1]:	https://github.com/dmlc/tvm/issues/151#issuecomment-303152024
[2]:	https://github.com/onnx/onnx
[3]:	https://github.com/dmlc/nnvm
[4]:	https://www.tensorflow.org/performance/xla/
[5]:	https://github.com/NervanaSystems/ngraph
[6]:	http://halide-lang.org/
[7]:	http://darkroom-lang.org/
[8]:	https://github.com/Maratyszcza/NNPACK
[9]:	https://developer.nvidia.com/cudnn
[10]:	https://github.com/botonchou/libdnn
[11]:	https://github.com/NervanaSystems/maxas
[12]:	https://github.com/NervanaSystems/maxas/blob/master/sgemm/sgemm_final_64.sass