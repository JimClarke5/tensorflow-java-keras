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
 * Computes the mean of absolute difference between labels and predictions.
 *
 * @author Jim Clarke
 */
public class MeanAbsoluteError extends MeanMetricWrapper implements LossFunction {

    public static final String DEFAULT_NAME = "mean_absolute_error";

    /**
     * Creates a MeanAbsoluteError
     */
    public MeanAbsoluteError(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }

    public MeanAbsoluteError(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }

    /**
     * Creates a MeanAbsoluteError
     *
     * @param name the name of the loss
     */
    public MeanAbsoluteError(Ops tf, String name) {
        this(tf, name, null);
    }

    public MeanAbsoluteError(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand losses = Losses.mean_absolute_error(tf, labels, predictions);
        return losses;
    }

}
