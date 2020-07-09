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
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;

/**
 * Computes root mean squared error metric between `y_true` and `y_pred`.
 * @author jbclarke
 */
public class RootMeanSquaredError extends Mean {
    public static final String DEFAULT_NAME = "root_mean_squared_error";
    
    /**
     * Create a RootMeanSquaredError metric.
     * 
     * @param tf The TensorFlow Ops
     */
    public RootMeanSquaredError(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }

    /**
     * Create a RootMeanSquaredError metric.
     * 
     * @param tf  The TensorFlow Ops
     * @param dType the type of the metric result.
     */
    public RootMeanSquaredError(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }

    /**
     * Create a RootMeanSquaredError metric.
     * 
     * @param tf  The TensorFlow Ops
     * @param name 
     */
    public RootMeanSquaredError(Ops tf, String name) {
        this(tf, name, null);
    }

    /**
     * Creates a RootMeanSquaredError Metric
     * 
     * @param tf the TensorFlow Ops
     * @param name
     * @param dType 
     */
    public RootMeanSquaredError(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Op updateState(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeights = args.length > 2 ? args[2] : null;
        
        yTrue = tf.dtypes.cast(yTrue, this.dType);
        yPred = tf.dtypes.cast(yPred, this.dType);
        
        Tuple ops = LossesImpl.squeezeOrExpandDimensions(tf, yPred, yTrue);
        yPred = ops.getPredictions();
        yTrue = ops.getLabels();
        Operand errorSquared = tf.math.squaredDifference(yPred, yTrue);
        
        return super.updateState(errorSquared, sampleWeights);
        
    }
    
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Operand result() {
        return tf.math.sqrt(tf.math.divNoNan(this.total, this.count));
    }
}
