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
import org.tensorflow.keras.metrics.impl.Reduce;
import org.tensorflow.op.Ops;

/**
 * Computes the (weighted) mean of the given values.
 * 
 * @author Jim Clarke
 */
public class Mean extends Reduce {

    /**
     * Creates a Mean metric.
     * 
     * @param tf  The TensorFlow Ops
     */
    public Mean(Ops tf) {
        this(tf, null, null);
    }

    /**
     * Creates a Mean metric.
     * 
     * @param tf The TensorFlow Ops
     * @param dType the type of the metric result.
     */
    public Mean(Ops tf, DataType dType) {
        this(tf, null, dType);
    }

    /**
     * Creates a Mean metric.
     * 
     * @param tf  The TensorFlow Ops
     * @param name name of the metric instance.
     */
    public Mean(Ops tf, String name) {
        this(tf, name, null);
    }

    
    /**
     * Creates a Mean metric.
     * 
     * @param tf  The TensorFlow Ops
     * @param name name of the metric instance.
     * @param dType the data type of the metric result.
     */
    public Mean(Ops tf, String name, DataType dType) {
        super(tf, name, Reduction.WEIGHTED_MEAN, dType);
    }

}
