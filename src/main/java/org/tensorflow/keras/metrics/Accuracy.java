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
import org.tensorflow.keras.backend.tf.Tuple;
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.keras.metrics.impl.MetricUtils;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Calculates how often predictions equals labels.
 * @author Jim Clarke
 */
public class Accuracy extends MeanMetricWrapper  implements LossFunction {
    
    public static final String DEFAULT_NAME = "accuracy";
    
    /**
     * Creates an Accuracy Metric
     * 
     * @param tf the TensorFlow Ops
     */
    public Accuracy(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }
    
    /**
     * Creates an Accuracy Metric
     * 
     * @param tf the TensorFlow Ops
     * @param dType  the data type for the metric
     */
    public Accuracy(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }
    
    /**
     * Creates an Accuracy Metric
     * 
     * @param tf the TensorFlow Ops
     * @param name the name of the metric
     */
    public Accuracy(Ops tf, String name) {
        this(tf, name, null);
    }
    
    /**
     * Creates an Accuracy Metric
     * 
     * @param tf the TensorFlow Ops
     * @param name the name of the metric
     * @param dType the data type for the metric
     */
    public Accuracy(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
        super.setLoss(this);
    }

    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        Tuple tuple = MetricUtils.raggedAssertCompatibleAndGetFlatValues(tf, labels, predictions);
        labels = tuple.getLabels();
        predictions = tuple.getPredictions();
        
        assert ShapeUtils.isCompatibleWith(predictions.asOutput().shape(), labels.asOutput().shape()) :
                String.format("Shapes %s and %s are incompatible", 
                        predictions.asOutput().shape().toString(),
                        labels.asOutput().shape().toString());
        if (labels.asOutput().dataType() != predictions.asOutput().dataType()) {
            predictions = tf.dtypes.cast(predictions, labels.asOutput().dataType());
        }
        return tf.dtypes.cast(tf.math.equal(labels, predictions), labels.asOutput().dataType());
    }
    
}
