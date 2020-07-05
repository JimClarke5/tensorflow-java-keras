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
 * Computes the hinge loss between labels and predictions.
 *
 * @author Jim Clarke
 */
public class Hinge extends Loss {

    /**
     * Creates a Hinge with Reduction.AUTO
     */
    public Hinge(Ops tf) {
        super(tf, "hinge");
    }

    /**
     * Creates a Hinge with Reduction.AUTO
     *
     * @param name the name of the loss
     */
    public Hinge(Ops tf, String name) {
        super(tf, name);
    }

    /**
     * Creates a Hinge
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public Hinge(Ops tf, Reduction reduction) {
        super(tf, "hinge", reduction);
    }

    /**
     * Creates a Hinge
     *
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public Hinge(Ops tf, String name, Reduction reduction) {
        super(tf, name, reduction);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.hinge(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
