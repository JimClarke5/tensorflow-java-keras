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
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.impl.MeanMetricWrapper;
import org.tensorflow.keras.metrics.impl.MetricsImpl;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Calculates how often predictions equals labels.
 *
 * @author Jim Clarke
 */
public class BinaryAccuracy extends MeanMetricWrapper implements LossFunction {

    public static final String DEFAULT_NAME = "binary_accuracy";
    public static final float DEFAULT_THRESHOLD = 0.5f;

    private final float threshold;

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     */
    public BinaryAccuracy(Ops tf) {
        this(tf, DEFAULT_NAME, DEFAULT_THRESHOLD, null);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param threshold
     */
    public BinaryAccuracy(Ops tf, float threshold) {
        this(tf, DEFAULT_NAME, threshold, null);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param dType the data type of the metric result
     */
    public BinaryAccuracy(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, DEFAULT_THRESHOLD, dType);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param name the name of the metric instance
     */
    public BinaryAccuracy(Ops tf, String name) {
        this(tf, name, DEFAULT_THRESHOLD, null);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param name the name of the metric instance
     * @param threshold
     */
    public BinaryAccuracy(Ops tf, String name, float threshold) {
        this(tf, name, threshold, null);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param name the name of the metric instance
     * @param dType the data type of the metric result
     */
    public BinaryAccuracy(Ops tf, String name, DataType dType) {
        this(tf, name, DEFAULT_THRESHOLD, dType);
    }

    /**
     * Creates a BinaryAccuracy metric
     *
     * @param tf the TensorFlow Ops
     * @param name the name of the metric instance
     * @param threshold a the threshold for deciding whether prediction values
     * are 1 or 0
     * @param dType the data type of the metric result
     */
    public BinaryAccuracy(Ops tf, String name, float threshold, DataType dType) {
        super(tf, name, dType);
        this.threshold = threshold;
        super.setLoss(this);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions, Operand<T> sampleWeights) {
        MetricsImpl.debug("labels", labels);
        MetricsImpl.debug("predictions", predictions);
        DataType dType = predictions.asOutput().dataType();
        Operand thresholdCast = tf.dtypes.cast(tf.constant(threshold), dType);
        predictions = tf.dtypes.cast(tf.math.greater(predictions, thresholdCast), dType);
        MetricsImpl.debug("predictions2", predictions);
        labels = tf.dtypes.cast(labels, dType);
        Operand result = tf.dtypes.cast(tf.math.equal(labels, predictions), predictions.asOutput().dataType());
        MetricsImpl.debug("result", result);
        return result;
    }

    /**
     * Gets the threshold value
     *
     * @return the threshold value
     */
    public double getThreshold() {
        return this.threshold;
    }

}
