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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.keras.backend.K;
import org.tensorflow.keras.backend.tf.Tuple;
import org.tensorflow.keras.backend.tf.WeightsBroadcastOps;
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Reduction;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;

/**
 * Encapsulates metrics that perform a reduce operation on the values.
 * 
 * @author Jim Clarke
 */
public abstract class Reduce extends Metric {

    public static final String TOTAL = "total";
    public static final String COUNT = "count";

    protected Variable<TFloat32> total;
    protected Variable<TFloat32> count;
    
    private final String totalName;
    private final String countName;

    protected final Reduction reduction;

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     */
    public Reduce(Ops tf) {
        this(tf, null, Reduction.SUM, null);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param reduction the type of reduction
     */
    public Reduce(Ops tf, Reduction reduction) {
        this(tf, null, reduction, null);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param reduction the type of reduction
     * @param dType the  data type of the metric result
     */
    public Reduce(Ops tf, Reduction reduction, DataType dType) {
        this(tf, null, reduction, dType);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param dType the  data type of the metric result
     */
    public Reduce(Ops tf, DataType dType) {
        this(tf, null, Reduction.SUM, dType);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param name the name of the metric instance.
     */
    public Reduce(Ops tf, String name) {
        this(tf, name, Reduction.SUM, null);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param name the name of the metric instance.
     * @param reduction the type of reduction 
     */
    public Reduce(Ops tf, String name, Reduction reduction) {
        this(tf, name, reduction, null);
    }

    /**
     * Creates a `Reduce` instance.
     * 
     * @param tf The TensorFlow Ops
     * @param name the name of the metric instance.
     * @param reduction the type of reduction
     * @param dType the  data type of the metric result
     */
    public Reduce(Ops tf, String name, Reduction reduction, DataType dType) {
        super(tf, name, dType);
        this.reduction = reduction;
        this.totalName = this.getVariableName(TOTAL);
        this.countName = this.getVariableName(COUNT);
        init();
    }

    /**
     * initialize the Variables
     */
    private void init() {
        Zeros zeros = new Zeros(tf);
        
        
        this.total = getVariable(getTotalName());
        if (this.total == null) {
            this.total = tf.withName(getTotalName()).variable(
                    zeros.call(tf.constant(Shape.scalar()), TFloat32.DTYPE));
            this.addVariable(getTotalName(), this.total, zeros);
        }
        if (reduction == Reduction.SUM_OVER_BATCH_SIZE || reduction == Reduction.WEIGHTED_MEAN) {
            this.count = getVariable(getCountName());
            if (getCount() == null) {
                 this.count = tf.withName(getCountName()).variable(
                    zeros.call(tf.constant(Shape.scalar()), TFloat32.DTYPE));
                this.addVariable(getCountName(), this.count, zeros);
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<Op> updateStateList(Operand... operands) {
        List<Op> updateOperations = new ArrayList<>();
        Operand values = operands[0];
        Operand sampleWeight = operands.length > 1 ? operands[1] : null;
        if (dType != null) {
            values = tf.dtypes.cast(values, dType);
        }
        DataType dtype = values.asOutput().dataType();
        
        if (sampleWeight != null) {
            sampleWeight = tf.dtypes.cast(sampleWeight, dtype);
            Tuple tuple = LossesImpl.squeezeOrExpandDimensions(tf, null, values, sampleWeight);
            values = tuple.getPredictions();
            sampleWeight = tuple.getSampleWeights();
            sampleWeight = WeightsBroadcastOps.broadcastWeights(tf, sampleWeight, values);
            values = tf.math.mul(values, sampleWeight);

        }

        Operand<TFloat32> valueSum = tf.dtypes.cast(tf.reduceSum(values, K.allAxis(tf, values)), TFloat32.DTYPE);
        
        Operand<TFloat32> totalUpdate = this.variableAssignAdd(this.getTotalName(), this.total, valueSum);
        updateOperations.add(totalUpdate);
        Operand<TFloat32> numValues;
        if (reduction != Reduction.SUM) {
            switch (reduction) {
                case SUM_OVER_BATCH_SIZE:
                    numValues = tf.dtypes.cast(
                            tf.constant(values.asTensor().shape().size()),
                            TFloat32.DTYPE);
                    break;
                case WEIGHTED_MEAN:
                    if (sampleWeight == null) {
                        numValues = tf.dtypes.cast(tf.constant(values.asOutput().shape().size()), TFloat32.DTYPE);
                    } else {
                        numValues = tf.reduceSum(sampleWeight, K.allAxis(tf, values));
                    }
                    break;
                default:
                    throw new UnsupportedOperationException(
                            String.format("reduction [%s] not implemented", reduction));
            }
            Op totalCount = this.variableAssignAdd(this.getCountName(), this.count, tf.dtypes.cast(numValues, TFloat32.DTYPE));
                    
                   // tf.assignAdd(getCount(), tf.dtypes.cast(numValues, TFloat32.DTYPE));
            updateOperations.add(totalCount);
        }
        
        return updateOperations;
        

    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Operand result(Ops rtf) {
        switch (this.reduction) {
            case SUM:
                return dType == null ? rtf.identity(this.getTotal()) : rtf.dtypes.cast(rtf.identity(this.getTotal()), dType);
            case WEIGHTED_MEAN:
            case SUM_OVER_BATCH_SIZE:
                Operand result = rtf.math.divNoNan(getTotal(), rtf.dtypes.cast(getCount(), getTotal().asOutput().dataType()));
                return dType == null ? result : rtf.dtypes.cast(result, dType);
            default:
                throw new UnsupportedOperationException(
                        String.format("reduction [%s] not implemented", reduction));
        }
    }

    protected void print(String prefix, Operand<TFloat32> operand) {
        if (tf.scope().env().isGraph()) {
            try (Session session = new Session((Graph) tf.scope().env())) {
                AtomicInteger index = new AtomicInteger();
                try (Tensor<TFloat32> result = session.runner().fetch(operand).run().get(0).expect(TFloat32.DTYPE)) {
                    if (result.data().size() > 1) {
                        result.data().scalars().forEach(f -> {
                            System.out.printf("%s: %d). %f\n", prefix, index.incrementAndGet(), f.getFloat());
                        });
                    } else {
                        System.out.printf("%s: %d). %f\n", prefix, index.incrementAndGet(), result.data().getFloat());
                    }
                }
            }
        }
    }

    /**
     * @return the reduction
     */
    public Reduction getReduction() {
        return reduction;
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

    /**
     * @return the totalName
     */
    public String getTotalName() {
        return totalName;
    }

    /**
     * @return the countName
     */
    public String getCountName() {
        return countName;
    }

    

}
