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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.backend.tf.Tuple;
import org.tensorflow.keras.backend.tf.WeightsBroadcastOps;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;

/**
 * Computes the element-wise (weighted) mean of the given tensors.
 * @author jbclarke
 */
public class MeanTensor extends Metric {
    public static final String DEFAULT_NAME = "mean_tensor";
    public static final String TOTAL = "total";
    public static final String COUNT = "count";
    
    private Shape shape;
    private Variable<TFloat32> total;
    private Variable<TFloat32> count;
    private boolean initialized;
    
    /**
     * create a metric with name = class name and reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     */
    public MeanTensor(Ops tf) {
        this(tf, DEFAULT_NAME, null);
    }

    /**
     * create a metric with reduction = AUTO
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param name the name of the metric
     */
    public MeanTensor(Ops tf, String name) {
        this(tf, name, null);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow Ops when using Eager Mode
     * @param dType the DataType to use
     */
    public MeanTensor(Ops tf, DataType dType) {
        this(tf, DEFAULT_NAME, dType);
    }

    /**
     * create a metric
     *
     * @param tf the TensorFlow ops
     * @param name the name of this metric
     * @param dType the DataType
     */
    public MeanTensor(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
    }
    
    private Op[] init(Shape shape) {
        Op[] initializers =null; 
        this.shape = shape;
        Zeros zeros = new Zeros(tf);
        
        total = getVariable(TOTAL);
        if (total == null) {
            total = tf.withName(TOTAL).variable(
                    zeros.call(tf.constant(shape), TFloat32.DTYPE));
            this.addVariable(TOTAL, total, zeros);
            if(initializers == null)
                initializers = new Op[2];
            initializers[0] = tf.assign(total, tf.zeros(tf.constant(shape), TFloat32.DTYPE));
        }
        count = getVariable(COUNT);
        if (count == null) {
             count = tf.withName(COUNT).variable(
                zeros.call(tf.constant(shape), TFloat32.DTYPE));
                this.addVariable(COUNT, count, zeros);
                if(initializers == null)
                    initializers = new Op[2];
                initializers[1] = tf.assign(count, tf.zeros(tf.constant(shape), TFloat32.DTYPE));
        }
        this.initialized = true;
        
        return initializers;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Op updateState(Operand... args) {
        Operand values = args[0];
        Operand sampleWeight = args.length > 1? args[1] : null;
        
        values = tf.dtypes.cast(values, TFloat32.DTYPE);
        
        Op[] initializers = null;
        
        if(!this.initialized) {
            initializers = init(values.asOutput().shape());
        }
        if(!this.shape.equals(values.asOutput().shape())) {
            throw new IllegalArgumentException(
                String.format("MeanTensor input values must always have the same shape. Expected shape (set during the first call): %s. Got %s",
                         this.shape.toString(),
                         values.asOutput().shape().toString())
            );
        }
        
        Operand numValues = tf.onesLike(values);
        if(sampleWeight != null) {
            sampleWeight = tf.dtypes.cast(sampleWeight, TFloat32.DTYPE);
            Tuple tuple = LossesImpl.squeezeOrExpandDimensions(tf, null, values, sampleWeight);
            values = tuple.getPredictions();
            sampleWeight = tuple.getSampleWeights();
            try {
                sampleWeight = WeightsBroadcastOps.broadcastWeights(tf, sampleWeight, values);
            }catch(IllegalArgumentException ex) {
                int ndim = values.asOutput().shape().numDimensions();
                int weightNdim = sampleWeight.asOutput().shape().numDimensions();
                int[] range = new int[ndim - weightNdim];
                for(int i = weightNdim; i < ndim; i++) {
                    range[i] = i;
                }
                values = tf.math.mean(values, tf.constant(range) );
            }
            numValues= tf.math.mul(numValues, sampleWeight);
            values = tf.math.mul(values, sampleWeight);
            
        }
        
        List<Op> controlOps = new ArrayList<>();
        Operand countAdd;
        Ops tf1 = initializers  != null?
            tf.withSubScope("count_initializer").withControlDependencies(Arrays.asList(initializers[1])):
             tf;
        controlOps.add(tf1.assignAdd(this.count, numValues));
        if(initializers  != null) {
            controlOps.add(initializers[0]);
        }
        Ops tf2 = tf.withSubScope("total_count").withControlDependencies(controlOps);
        return tf2.assignAdd(this.total, values);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand result() {
        if(!this.initialized) {
            throw new IllegalStateException("MeanTensor does not have any result yet. Please  use `.update_state(value)` before retrieving the result.");
        }
        return tf.math.divNoNan(total, count);
    }

    /**
     * @return the total
     */
    public Variable<TFloat32> getTotal() {
        return total;
    }

    /**
     * @return the count
     */
    public Variable<TFloat32> getCount() {
        return count;
    }
}
