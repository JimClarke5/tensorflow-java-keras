/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.utils;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.NdArray;
import org.tensorflow.tools.ndarray.NdArrays;

/**
 *  TODO NDArray Utilities use in the Callbacks, this should be a part of NDArray
 * 
 * @author Jim Clarke
 */
public class ND {

    public static String toString(NdArray<?> array) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        AtomicBoolean first = new AtomicBoolean(true);
        array.elements(0).forEachIndexed((idx, v) -> {
            if (!first.get()) {
                sb.append(", ");
            } else {
                first.set(false);
            }
            Object f = v.getObject();
            if (v.rank() == 0) {
                sb.append(f);
            } else {
                sb.append(toString(v));
            }

        });
        sb.append("]");
        return sb.toString();
    }

    private static long[] getCoordinates(Shape shape, long index) {
        long[] coordinates = new long[shape.numDimensions()];

        int numDims = shape.numDimensions();
        int i = numDims - 1;
        for (; i >= 0; i--) {
            long size = shape.size(i);
            long mod = index % size;
            coordinates[i] = mod;
            index -= mod;

        }
        //coordinates[i] = index;
        return coordinates;

    }

    public static FloatNdArray sqrt(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.sqrt(v.getFloat()), idx);
        });
        return result;
    }

    public static FloatNdArray square(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() * v.getFloat(), idx);
        });
        return result;
    }

    public static FloatNdArray add(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() + b.getFloat(idx), idx);
        });
        return result;
    }

    public static FloatNdArray add(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());

        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() + scalar, idx);
        });
        return result;
    }

    public static FloatNdArray add(float scalar, FloatNdArray a) {
        return add(a, scalar);
    }

    public static FloatNdArray sub(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() - b.getFloat(idx), idx);
        });
        return result;
    }

    public static FloatNdArray sub(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() - scalar, idx);
        });
        return result;
    }

    public static FloatNdArray sub(float scalar, FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(scalar - v.getFloat(), idx);
        });
        return result;
    }

    public static FloatNdArray mul(FloatNdArray a, FloatNdArray b) {
        assert a.shape().equals(b.shape()) :
                String.format("ValueError: operands do not have same shapes %s %s ", a.shape(), b.shape());
        boolean sameSize = a.shape().size() == b.shape().size();
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        print("a", a);
        print("result", result);

        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            if (sameSize) {
                result.setFloat(v.getFloat() * b.getFloat(idx), idx);
            } else {
                float value = v.getFloat() * b.getFloat(idx[0], 0L);
                result.setFloat(value, idx);
            }
        });
        return result;
    }

    public static FloatNdArray mul(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() * scalar, idx);
        });
        return result;
    }

    public static FloatNdArray mul(float scalar, FloatNdArray a) {
        return mul(a, scalar);
    }

    public static FloatNdArray div(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() / b.getFloat(idx), idx);
        });
        return result;
    }

    public static FloatNdArray div(FloatNdArray a, float scalar) {
        assert (scalar != 0);
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat(v.getFloat() / scalar, idx);
        });
        return result;
    }

    public static FloatNdArray div(float scalar, FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            float value = v.getFloat() == 0.0F ? Float.NaN
                    : scalar / v.getFloat();
            result.setFloat(value, idx);
        });
        return result;
    }

    public static FloatNdArray pow(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.pow(v.getFloat(), b.getFloat(idx)), idx);
        });
        return result;
    }

    public static FloatNdArray pow(FloatNdArray a, float scalar) {
        assert (scalar != 0);
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.pow(v.getFloat(), scalar), idx);
        });
        return result;
    }

    public static FloatNdArray pow(float scalar, FloatNdArray a) {
        assert (scalar != 0);
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.pow(scalar, v.getFloat()), idx);
        });
        return result;
    }

    public static float[] flatten(FloatNdArray a) {
        float[] result = new float[(int) a.shape().size()];
        int nDims = a.shape().numDimensions();
        AtomicInteger counter = new AtomicInteger();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result[counter.getAndAdd(1)] = v.getFloat();
        });
        return result;
    }

    public static FloatNdArray max(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.max(v.getFloat(), b.getFloat(idx)), idx);
        });
        return result;
    }

    public static FloatNdArray max(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.max(v.getFloat(), scalar), idx);
        });
        return result;
    }

    public static FloatNdArray max(float scalar, FloatNdArray a) {
        return max(a, scalar);
    }

    public static FloatNdArray min(FloatNdArray a, FloatNdArray b) {
        assert (a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.min(v.getFloat(), b.getFloat(idx)), idx);
        });
        return result;
    }

    public static FloatNdArray min(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.min(v.getFloat(), scalar), idx);
        });
        return result;
    }

    public static FloatNdArray min(float scalar, FloatNdArray a) {
        return min(a, scalar);
    }

    public static FloatNdArray abs(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        int nDims = a.shape().numDimensions();
        a.elements(nDims - 1).forEachIndexed((idx, v) -> {
            result.setFloat((float) Math.abs(v.getFloat()), idx);
        });
        return result;
    }

    public static FloatNdArray sum(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        float sum = 0;
        for (int i = 0; i < a.size(); i++) {
            sum += a.getFloat(i);
        }
        return NdArrays.scalarOf(sum);
    }

    public static FloatNdArray sum(FloatNdArray a, int axis) {
        return sum(a, new Integer[]{axis}, false);
    }

    public static FloatNdArray sum(FloatNdArray a, int axis, boolean keepDims) {
        return sum(a, new Integer[]{axis}, keepDims);
    }

    public static FloatNdArray sum(FloatNdArray a, Integer[] axis, boolean keepDims) {
        Shape shape = a.shape();
        if (axis == null) {
            FloatNdArray result = sum(a);
            if (keepDims) {
                float scalar = result.getFloat(0);
                long[] dims = {1, 1};
                Shape bShape = Shape.of(dims);
                FloatNdArray resultK = NdArrays.ofFloats(bShape);
                resultK.setFloat(scalar, 0, 0);
                return resultK;
            }
            return result;
        } else if (axis.length == 1) {
            int nDims = shape.numDimensions();
            if (axis[0] < 0) {
                axis[0] = nDims + axis[0];
            }

            final float[] sums = new float[(int) shape.size(axis[0])];

            a.elements(nDims - 1).forEachIndexed((idx, v) -> {
                System.out.println(Arrays.toString(idx));
                sums[(int) idx[axis[0]]] += v.getFloat();
            });

            if (keepDims) {
                long[] newDims = shape.asArray();
                newDims[axis[0]] = 1;
                final AtomicInteger counter = new AtomicInteger();
                FloatNdArray arrayK = NdArrays.ofFloats(Shape.of(newDims));
                arrayK.elements(newDims.length - 1).forEachIndexed((idx, v) -> {
                    v.setFloat(sums[counter.getAndAdd(1)]);
                });
                return arrayK;
            } else {
                return NdArrays.vectorOf(sums);
            }
        } else {

            throw new UnsupportedOperationException("Multi Axis Not implemented Yet");

        }
    }

    public static FloatNdArray l2_norm(FloatNdArray x) {
        return l2_norm(x, -1);
    }

    public static FloatNdArray l2_norm(FloatNdArray x, int axis) {
        float epsilon = 1e-12F;
        FloatNdArray square_sum = ND.sum(ND.square(x), axis, true);
        FloatNdArray x_inv_norm = ND.div(1, ND.sqrt(ND.max(square_sum, epsilon)));
        return ND.mul(x, x_inv_norm);
    }

    public static void print(FloatNdArray a) {
        System.out.println("Shape: " + a.shape());
        a.elements(a.shape().numDimensions() - 1).forEachIndexed((idx, v) -> {
            System.out.printf("%s == %f\n", Arrays.toString(idx), a.getFloat(idx));
        });
        System.out.println();
    }

    public static void print(String header, FloatNdArray a) {
        System.out.print(header);
        System.out.print(" : ");

        print(a);
    }

    public static FloatNdArray create(float[] y, Shape shape) {
        FloatNdArray result = NdArrays.ofFloats(shape);
        AtomicInteger index = new AtomicInteger();
        result.elements(shape.numDimensions() - 1).forEachIndexed((idx, v) -> {
            v.setFloat(y[index.getAndAdd(1)]);
        });
        return result;
    }

}
