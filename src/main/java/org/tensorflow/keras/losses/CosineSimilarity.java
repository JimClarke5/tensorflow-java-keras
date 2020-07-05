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
 * Computes the categorical hinge loss between labels and [redictions.
 *
 * @author Jim Clarke
 */
public class CosineSimilarity extends Loss {

    private final int axis;

    /**
     * Creates a CosineSimilarity with Reduction.AUTO and axis=-1
     */
    public CosineSimilarity(Ops tf) {
        this(tf, "cosine_similarity", Reduction.AUTO, -1);
    }

    /**
     * Creates a CosineSimilarity with Reduction.AUTO
     *
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, int axis) {
        this(tf, "cosine_similarity", Reduction.AUTO, axis);
    }

    /**
     * Creates a CosineSimilarity with Reduction.AUTO and axis=-1
     *
     * @param name the name of the loss
     */
    public CosineSimilarity(Ops tf, String name) {
        this(tf, name, Reduction.AUTO, -1);
    }

    /**
     * Creates a CosineSimilarity with Reduction.AUTO
     *
     * @param name the name of the loss
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, String name, int axis) {
        this(tf, name, Reduction.AUTO, axis);
    }

    /**
     * Creates a CosineSimilarity and axis=-1
     *
     * @param reduction Type of Reduction to apply to loss.
     */
    public CosineSimilarity(Ops tf, Reduction reduction) {
        this(tf, "cosine_similarity", reduction, -1);
    }

    /**
     * Creates a CosineSimilarity
     *
     * @param reduction Type of Reduction to apply to loss.
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, Reduction reduction, int axis) {
        this(tf, "cosine_similarity", reduction, axis);
    }

    /**
     * Creates a CosineSimilarity and axis=-1
     *
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     */
    public CosineSimilarity(Ops tf, String name, Reduction reduction) {
        this(tf, name, reduction, -1);
    }

    /**
     * Creates a CosineSimilarity
     *
     * @param name the name of the loss
     * @param reduction Type of Reduction to apply to loss.
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, String name, Reduction reduction, int axis) {
        super(tf, name, reduction);
        this.axis = axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.cosine_similarity(tf, labels, predictions);
        return super.computeWeightedLoss(losses, getReduction(), sampleWeights);
    }

}
