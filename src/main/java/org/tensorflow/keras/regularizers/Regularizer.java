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
package org.tensorflow.keras.regularizers;

import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TType;

/**
 * Regularizer base class.
 * 
 * <p>Regularizers allow you to apply penalties on layer parameters or layer activity during optimization. These penalties are summed into the loss function that the network optimizes.
 */
public abstract class Regularizer {
    public static final float DEFAULT_REGULARIZATION_PENALTY = 0.01f;
    
    
    protected Ops tf;
    
    /**
     * Create a Regularizer
     * @param tf the TensorFlow ops.
     */
    protected Regularizer(Ops tf) {
        this.tf = tf;
    }
    
    /**
     * Compute a regularization penalty from an input.
     *
     * @param input teh weighted input
     * @param <T> the data type of the Operand
     * @return the result of computing the regularization penalty
     * 
     */
    public abstract <T extends TType> Operand<T> call(Operand<T> input);
}
