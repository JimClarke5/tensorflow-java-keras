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
 * Computes the Poisson loss between the labels and predictions
 *
 * @author Jim Clarke
 */
public class SquaredHinge extends Loss {

    /**
     * Creates a Poisson with Reduction.AUTO
     */
    public SquaredHinge(Ops tf) {
        super(tf, "squared_hinge");
    }

    /**
     * Creates a Poisson
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(Ops tf, Reduction reduction) {
        super(tf, "squared_hinge", reduction);
    }

    /**
     * Creates a Poisson
     *
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public SquaredHinge(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.squared_hinge(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
