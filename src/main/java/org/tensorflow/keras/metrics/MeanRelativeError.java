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
import org.tensorflow.keras.backend.tf.ConfusionMatrix;
import org.tensorflow.keras.backend.tf.Tuple;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.keras.utils.ShapeUtils;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;

/**
 *
 * @author jbclarke
 */
public class MeanRelativeError extends Mean{
    private  Operand  normalizer;
    
    /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, float[] normalizer) {
        this(tf, null, tf.constant(normalizer), null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, String name, float[] normalizer) {
        this(tf, name,  tf.constant(normalizer), null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     * @param dType the DataType to use
     */
    protected MeanRelativeError(Ops tf, float[] normalizer, DataType dType) {
        this(tf, null,  tf.constant(normalizer), dType);
    }
    
     /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, double[] normalizer) {
        this(tf, null, tf.constant(normalizer), null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, String name, double[] normalizer) {
        this(tf, name,  tf.constant(normalizer), null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     * @param dType the DataType to use
     */
    protected MeanRelativeError(Ops tf, double[] normalizer, DataType dType) {
        this(tf, null,  tf.constant(normalizer), dType);
    }
    
     /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, Operand normalizer) {
        this(tf, null, normalizer, null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     * @param normalizer The normalizer values with same shape as predictions.
     */
    protected MeanRelativeError(Ops tf, String name, Operand normalizer) {
        this(tf, name,  normalizer, null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param normalizer The normalizer values with same shape as predictions.
     * @param dType the DataType to use
     */
    protected MeanRelativeError(Ops tf, Operand normalizer, DataType dType) {
        this(tf, null,  normalizer, dType);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow ops
     * @param name the name of this metric
     * @param normalizer The normalizer values with same shape as predictions.
     * @param dType the DataType
     */
    protected MeanRelativeError(Ops tf, String name, Operand normalizer,  DataType dType) {
        super(tf, name, dType);
        this.normalizer = tf.dtypes.cast(normalizer, this.dType);
    }


    @Override
    public Op updateState(Operand... args) {
        Operand yTrue = args[0];
        Operand yPred = args[1];
        Operand sampleWeight = args.length > 2 ? args[2] : null;
        
        yTrue = tf.shape.flatten(tf.dtypes.cast(yTrue, this.dType));
        yPred = tf.shape.flatten(tf.dtypes.cast(yPred, this.dType));
        Tuple ops = LossesImpl.squeezeOrExpandDimensions(tf, yPred, yTrue, null);
        yPred = ops.getPredictions();
        yTrue = ops.getLabels();
        if(sampleWeight != null) {
            sampleWeight = tf.shape.flatten(sampleWeight);
        }
        
        
        
        Tuple tuple = ConfusionMatrix.removeSqueezableDimensions(tf, this.getNormalizer(), yPred);
        this.normalizer = tuple.getLabels();
        yPred = tuple.getPredictions();
        
        if(!ShapeUtils.isCompatibleWith(
                    yPred.asOutput().shape(), 
                    yTrue.asOutput().shape())) {
                throw new IllegalArgumentException(
                        String.format("Prediction shape %s is not compatible with labels shape %s",
                                yPred.asOutput().shape(),
                                yTrue.asOutput().shape()));
        }
        
        Operand relativeErrors = tf.math.divNoNan(tf.math.abs(tf.math.sub(yTrue, yPred)), this.getNormalizer());
        
        return super.updateState(relativeErrors, sampleWeight);
        
    }


    /**
     * @return the normalizer
     */
    public Operand getNormalizer() {
        return normalizer;
    }
}
