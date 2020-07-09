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
 * Computes the (weighted) sum of the given values.
 * 
 * @author jbclarke
 */
public class Sum extends Reduce {
    
    
    /**
     * Creates a `Sum` instance.
     * 
     * @param tf The TensorFlow Ops
     */
    public Sum(Ops tf) {
        this(tf, null, null);
    }

  
    /**
     * Creates a `Sum` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param dType the  data type of the metric result
     */
    public Sum(Ops tf, DataType dType) {
        this(tf, null, dType);
    }

    /**
     * Creates a `Sum` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param name the name of the metric instance.
     */
    public Sum(Ops tf, String name) {
        this(tf, name, null);
    }

    

    /**
     * Creates a `Sum` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param name the name of the metric instance.
     * @param dType the  data type of the metric result
     */
    public Sum(Ops tf, String name, DataType dType) {
        super(tf, name, Reduction.SUM, dType);
    }
    
}
