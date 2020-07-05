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
 * Computes the cross-entropy loss between true labels and predicted labels.
 *
 * @author Jim Clarke
 */
public class CategoricalCrossentropy extends MeanMetricWrapper implements LossFunction {

    public static final String DEFAULT_NAME = "categorical_crossentropy";

    private final boolean fromLogits;
    private final float labelSmoothing;
    private int axis;

    /**
     * Creates a SparseCategoricalCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     */
    public CategoricalCrossentropy(Ops tf) {
        this(tf, DEFAULT_NAME, false, 0.0F, -1, null);
    }

    public CategoricalCrossentropy(Ops tf, int axis) {
        this(tf, DEFAULT_NAME, false, 0.0F, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy labelSmoothing=0.0 and
     * fromLogits=false.
     *
     * @param name the name of this loss function
     */
    public CategoricalCrossentropy(Ops tf, String name) {
        this(tf, name, false, 0.0F, -1, null);
    }

    public CategoricalCrossentropy(Ops tf, String name, int axis) {
        this(tf, name, false, 0.0F, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public CategoricalCrossentropy(Ops tf, boolean fromLogits) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, -1, null);
    }

    public CategoricalCrossentropy(Ops tf, boolean fromLogits, int axis) {
        this(tf, DEFAULT_NAME, fromLogits, 0.0F, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy ,
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     */
    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits) {
        this(tf, name, fromLogits, 0.0F, -1, null);
    }

    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits, int axis) {
        this(tf, name, fromLogits, 0.0F, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy ,
     *
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public CategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, -1, null);
    }

    public CategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, int axis) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, axis, null);
    }

    public CategoricalCrossentropy(Ops tf, boolean fromLogits, float labelSmoothing, int axis, DataType dType) {
        this(tf, DEFAULT_NAME, fromLogits, labelSmoothing, axis, null);
    }

    /**
     * Creates a SparseCategoricalCrossentropy
     *
     * @param name the name of this loss function
     * @param fromLogits Whether to interpret yPred as a tensor of logit values
     * @param labelSmoothing Float in [0, 1]. When 0, no smoothing occurs. When
     * > 0, we compute the loss between the predicted labels and a smoothed
     * version of the true labels, where the smoothing squeezes the labels
     * towards 0.5. Larger values of label_smoothing correspond to heavier
     * smoothing.
     */
    public CategoricalCrossentropy(Ops tf, String name, boolean fromLogits, float labelSmoothing, int axis, DataType dType) {
        super(tf, name, dType);
        this.fromLogits = fromLogits;
        this.labelSmoothing = labelSmoothing;
        this.axis = axis;
        super.setLoss(this);
    }

    /**
     * @return the fromLogits
     */
    public boolean isFromLogits() {
        return fromLogits;
    }

    /**
     * @return the labelSmoothing
     */
    public float getLabelSmoothing() {
        return labelSmoothing;
    }

    /**
     * @return the axis
     */
    public float getAxis() {
        return axis;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.categorical_crossentropy(tf, labels, predictions, isFromLogits(), getLabelSmoothing(), axis);
        return losses;
    }

}
