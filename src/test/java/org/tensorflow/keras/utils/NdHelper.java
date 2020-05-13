/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import org.tensorflow.tools.ndarray.FloatNdArray;
import org.tensorflow.tools.ndarray.NdArrays;

/**
 *
 * @author Jim Clarke
 */
public class NdHelper {
    
     public static FloatNdArray sqrt(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat((float)Math.sqrt(a.getFloat(i)), i);
        }
        return result;
    }
     
      public static FloatNdArray squared(FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            float f = a.getFloat(i);
            float sqr = f*f;
            result.setFloat(sqr, i);
        }
        return result;
    }
     
    public static FloatNdArray add(FloatNdArray a, FloatNdArray b) {
        assert(a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) + b.getFloat(i), i);
        }
        return result;
    }
    
    public static FloatNdArray add(FloatNdArray a,float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) + scalar, i);
        }
        return result;
    }
    public static FloatNdArray add(float scalar, FloatNdArray a) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) + scalar, i);
        }
        return result;
    }
    
    
    public static FloatNdArray minus(FloatNdArray a, FloatNdArray b) {
        assert(a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) - b.getFloat(i), i);
        }
        return result;
    }
    
    public static FloatNdArray minus(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) - scalar, i);
        }
        return result;
    }
    
    public static FloatNdArray minus(float scalar, FloatNdArray a ) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(scalar - a.getFloat(i), i);
        }
        return result;
    }
    
    public static FloatNdArray mul(FloatNdArray a, FloatNdArray b) {
        assert(a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) * b.getFloat(i), i);
        }
        return result;
    }
    
    public static FloatNdArray mul(FloatNdArray a, float scalar) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) * scalar, i);
        }
        return result;
    }
    
    public static FloatNdArray mul(float scalar, FloatNdArray a ) {
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) * scalar, i);
        }
        return result;
    }
    
    public static FloatNdArray div(FloatNdArray a, FloatNdArray b) {
        assert(a.shape().size() == b.shape().size());
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) / b.getFloat(i), i);
        }
        return result;
    }
    
    public static FloatNdArray div(FloatNdArray a, float scalar) {
        assert(scalar != 0);
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(a.getFloat(i) / scalar, i);
        }
        return result;
    }
    
    public static FloatNdArray div( float scalar, FloatNdArray a) {
        assert(scalar != 0);
        FloatNdArray result = NdArrays.ofFloats(a.shape());
        for(int i = 0; i < a.size(); i++) {
            result.setFloat(scalar / a.getFloat(i) , i);
        }
        return result;
    }
    
    public static float[] toArray(FloatNdArray a){
        float[] result = new float[(int)a.shape().size()];
        for(int i = 0; i < result.length; i++) {
            result[i] = a.getFloat(i);
        }
        return result;
    }
    
    public static void print(FloatNdArray a) {
        for(FloatNdArray v : a.scalars()) {
            System.out.printf("%f, ", v.getFloat());
        }
        System.out.println();
    }
    
}
