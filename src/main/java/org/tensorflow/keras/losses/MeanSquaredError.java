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
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the mean of squares of errors between labels and predictions.
 *
 * @author Jim Clarke
 */
public class MeanSquaredError extends Loss {

    /**
     * Creates a MeanSquaredError with Reduction.AUTO
     */
    public MeanSquaredError(Ops tf) {
        super(tf, "mean_squared_error");
    }

    /**
     * Creates a MeanSquaredError
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredError(Ops tf, Reduction reduction) {
        super(tf, "mean_squared_error", reduction);
    }

    /**
     * Creates a MeanSquaredError with Reduction.AUTO
     *
     * @param name the name for the loss function
     */
    public MeanSquaredError(Ops tf, String name) {
        super(tf, name);
    }

    /**
     * Creates a MeanSquaredError
     *
     * @param name the name for the loss function
     * @param reduction Type of Reduction to apply to loss.
     */
    public MeanSquaredError(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_squared_error(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
