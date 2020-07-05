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
import org.tensorflow.keras.losses.*;
import org.tensorflow.Operand;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Computes the categorical hinge loss between labels and [redictions.
 *
 * @author Jim Clarke
 */
public class CosineSimilarity extends MeanMetricWrapper implements LossFunction {

    public static final String DEFAULT_NAME = "cosine_similarity";

    private final int axis;

    /**
     * Creates a CosineSimilarity and axis=-1
     */
    public CosineSimilarity(Ops tf) {
        this(tf, DEFAULT_NAME, -1, null);
    }

    public CosineSimilarity(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, -1, dType);
    }

    /**
     * Creates a CosineSimilarity
     *
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, int axis) {
        this(tf, DEFAULT_NAME, axis, null);
    }

    public CosineSimilarity(Ops tf, int axis, DataType dType) {
        this(tf, DEFAULT_NAME, axis, dType);
    }

    /**
     * Creates a CosineSimilarity and axis=-1
     *
     * @param name the name of the loss
     */
    public CosineSimilarity(Ops tf, String name) {
        this(tf, name, -1, null);
    }

    public CosineSimilarity(Ops tf, String name, DataType dType) {
        this(tf, name, -1, dType);
    }

    /**
     * Creates a CosineSimilarity
     *
     * @param name the name of the loss
     * @param axis The dimension along which the cosine similarity is computed.
     */
    public CosineSimilarity(Ops tf, String name, int axis, DataType dType) {
        super(tf, name, dType);
        this.axis = axis;
        super.setLoss(this);
    }

    public int getAxis() {
        return axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        // Losses.cosine_similarity does a tf.neg() on the result, Metrics version does not.
        //Operand losses = Losses.cosine_similarity(tf, labels, predictions);
        Operand losses = Metrics.cosine_proximity(tf, labels, predictions, axis);
        return losses;
    }

}
