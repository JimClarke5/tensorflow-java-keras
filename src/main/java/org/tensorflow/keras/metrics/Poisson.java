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
 * Computes the Poisson loss between the labels and predictions
 *
 * @author Jim Clarke
 */
public class Poisson extends MeanMetricWrapper implements LossFunction {

    public static final String DEFAULT_NAME = "poisson";

    /**
     * Creates a Poisson
     *
     * @param tf the TensorFlow Ops
     */
    public Poisson(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }

    public Poisson(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }

    /**
     * Creates a Poisson
     *
     * @param tf the TensorFlow Ops
     * @param name the name of the loss
     */
    public Poisson(Ops tf, String name) {
        this(tf, name, null);
    }

    public Poisson(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        return Losses.poisson(tf, labels, predictions);
    }

}
