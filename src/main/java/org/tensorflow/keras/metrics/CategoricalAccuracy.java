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
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.op.Ops;
import org.tensorflow.op.math.Equal;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.family.TNumber;

/**
 * Calculates how often predictions equals labels.
 * @author Jim Clarke
 */
public class CategoricalAccuracy extends MeanMetricWrapper  implements LossFunction {
    
    public static final String DEFAULT_NAME = "categorical_accuracy";
    
    public CategoricalAccuracy(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }
    
    public CategoricalAccuracy(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }
    public CategoricalAccuracy(Ops tf, String name) {
        this(tf, name, null);
    }
    
    public CategoricalAccuracy(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }

    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Operand trueMax = tf.math.argMax(labels, K.minusOne(tf));
        Operand predMax = tf.math.argMax(predictions, K.minusOne(tf));
        Equal equals = tf.math.equal(trueMax, predMax);
        return tf.dtypes.cast(equals, labels.asOutput().dataType());
    }
    
}
