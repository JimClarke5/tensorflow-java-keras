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
package org.tensorflow.keras.metrics.impl;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.Mean;
import org.tensorflow.keras.metrics.Metrics;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class MeanMetricWrapper extends Mean {

    protected LossFunction loss;

    public MeanMetricWrapper(Ops tf) {
        this(tf, null, null);
    }

    public MeanMetricWrapper(Ops tf, String name) {
        this(tf, name, null);
    }

    public MeanMetricWrapper(Ops tf, DataType dType) {
        this(tf, null, dType);
    }

    public MeanMetricWrapper(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
    }

    public final void setLoss(LossFunction loss) {
        this.loss = loss;
    }

    @Override
    public Op updateState(Operand... operands) {
        Operand labels = operands[0];
        Operand predictions = operands[1];
        Operand sampleWeights = operands.length > 2 ? operands[2] : null;
        Operand losses = loss.call(labels, predictions, null);
        MetricsImpl.debug("losses", losses);
        return super.updateState(losses, sampleWeights);
    }

}
