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
package org.tensorflow.keras.losses;

import org.tensorflow.Operand;
import org.tensorflow.types.family.TNumber;

/**
 * The functional interface for all Loss Functions
 *
 * @author Jim Clarke
 */
@FunctionalInterface
public interface LossFunction {

    /**
     * Calculates the loss
     *
     * @param <T> Operands extend TNumber
     * @param tf the TensorFlow Ops
     * @param labels the truth values or labels
     * @param predictions the predictions
     * @param sampleWeights Optional sample_weight acts as a coefficient for the
     * loss. If a scalar is provided, then the loss is simply scaled by the
     * given value. If sample_weight is a tensor of size [batch_size], then the
     * total loss for each sample of the batch is rescaled by the corresponding
     * element in the sample_weight vector. If the shape of sample_weight is
     * [batch_size, d0, .. dN-1] (or can be broadcasted to this shape), then
     * each loss element of y_pred is scaled by the corresponding value of
     * sample_weight. (Note on dN-1: all loss functions reduce by 1 dimension,
     * usually axis=-1.)
     * @return the loss
     */
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights);
}
