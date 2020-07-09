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
package org.tensorflow.keras.metrics;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes how often integer targets are in the top K predictions.
 * 
 * @author jbclarke
 */
public class SparseTopKCategoricalAccuracy extends MeanMetricWrapper  implements LossFunction {
    public static final String DEFAULT_NAME = "parse_top_k_categorical_accuracy";
    public static final int DEFAULT_K = Metrics.DEFAULT_K;
    
    private final int k;
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     *
     * @param tf the TensorFlow Ops
     */
    public SparseTopKCategoricalAccuracy(Ops tf) {
        this(tf, DEFAULT_NAME, DEFAULT_K, null);
    }
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     *
     * @param tf the TensorFlow Ops
     */
    public SparseTopKCategoricalAccuracy(Ops tf, int k) {
        this(tf, DEFAULT_NAME, k, null);
    }

    /**
     * Creates a SparseTopKCategoricalAccuracy
     * 
     * @param tf the TensorFlow Ops
     * @param dType the data type of the metric result, defaults to TFloat32 
     */
    public SparseTopKCategoricalAccuracy(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, DEFAULT_K, dType);
    }
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     * 
     * @param tf the TensorFlow Ops
     * @param k Number of top elements to look at for computing accuracy.
     * Defaults to 5.
     * @param dType the data type of the metric result, defaults to TFloat32  
     */
    public SparseTopKCategoricalAccuracy(Ops tf, int k, DataType dType) {
        this(tf, DEFAULT_NAME, k, dType);
    }

    /**
     * Creates a SparseTopKCategoricalAccuracy
     *
     * @param tf the TensorFlow Ops
     * @param name the name for the metric instance
     */
    public SparseTopKCategoricalAccuracy(Ops tf, String name) {
        this(tf, name, DEFAULT_K, null);
    }
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     *
     * @param tf the TensorFlow Ops
     * @param name the name for the metric instance
     * @param dType the data type of the metric result, defaults to TFloat32  
     */
    public SparseTopKCategoricalAccuracy(Ops tf, String name, DataType dType) {
        this(tf, name, DEFAULT_K, dType);
    }
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     * 
     * @param tf the TensorFlow Ops
     * @param name the name for the metric instance
     * @param k Number of top elements to look at for computing accuracy.
     * Defaults to 5. 
     */
    public SparseTopKCategoricalAccuracy(Ops tf, String name, int k) {
        this(tf, name, k, null);
    }
    
    /**
     * Creates a SparseTopKCategoricalAccuracy
     * 
     * @param tf the TensorFlow Ops
     * @param name the name for the metric instance
     * @param k  Number of top elements to look at for computing accuracy.
     * Defaults to 5.
     * @param dType the data type of the metric result, defaults to TFloat32   
     */

    public SparseTopKCategoricalAccuracy(Ops tf, String name, int k, DataType dType) {
        super(tf, name, dType);
        this.k = k;
        super.setLoss(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Metrics.sparse_top_k_categorical_accuracy(tf, labels, predictions, k);
        return losses;
    }
    
    /**
     * @return the k
     */
    public int getK() {
        return k;
    }
}
