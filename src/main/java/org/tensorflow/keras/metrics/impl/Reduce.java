/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
import org.tensorflow.keras.initializers.Zeros;
import org.tensorflow.keras.metrics.Metric;
import org.tensorflow.keras.metrics.Reduction;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Assign;
import org.tensorflow.op.core.NoOp;
import org.tensorflow.op.core.Variable;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;

/**
 *
 * @author Jim Clarke
 */
public class Reduce extends Metric {
    
    public static final String TOTAL = "total";
    public static final String COUNT = "count";

    private Variable<TFloat32> total;
    private Variable<TFloat32> count;

    public Reduce(Ops tf, Reduction reduction) {
        this(tf, null, reduction, null);
    }
    
    public Reduce(Ops tf, Reduction reduction, DataType dType) {
        this(tf, null, reduction, dType);
    }
    
    public Reduce(Ops tf, String name, Reduction reduction, DataType dType) {
        super(tf, name, reduction, dType);
        init();
    }

    
    private void init() {
        total = tf.withName(TOTAL).variable(Shape.scalar(), TFloat32.DTYPE);
        Assign<TFloat32> totalInit = tf.assign(total, tf.constant(0.F));
        this.addVariable(TOTAL, total, new Zeros(tf));
        if(graph != null)
            graph.addInitializer(totalInit);
        if (reduction == Reduction.SUM_OVER_BATCH_SIZE || reduction == Reduction.WEIGHTED_MEAN) {
            count = tf.withName(COUNT).variable(Shape.scalar(), TFloat32.DTYPE);
            Assign<TFloat32> countInit = tf.assign(count, tf.constant(0f));
            this.addVariable(COUNT, count, new Zeros(tf));
            if(graph != null) 
                graph.addInitializer(countInit);
        }
    }
        

    @Override
    public Op updateState(Operand... operands) {
        Operand values = operands[0];
        Operand sampleWeight = operands[1];
        if (dType != null) {
            values = tf.dtypes.cast(values, dType);
        }
        DataType dtype = values.asOutput().dataType();
        List<Op> updateOperations = new ArrayList<>();
        if (sampleWeight != null) {
            //print("value b4 sampleWeight",(Operand<TFloat32>)values);
            sampleWeight = tf.dtypes.cast(sampleWeight, dtype);
            values = tf.math.mul(values, sampleWeight);

            // TODO ???
            /**
             * ************************
             * int ndims = values.asOutput().shape().numDimensions(); int
             * weightNdim = sampleWeight.asOutput().shape().numDimensions();
             * int[] axis = new int[ndims - weightNdim]; for(int i = weightNdim;
             * i < ndims; i++) axis[i] = i; if(reduction == Reduction.SUM) {
             * values = tf.reduceSum(values, tf.constant(axis)); }else { values
             * = tf.math.mean(values, tf.constant(axis)); }
            * ************
             */
        }
        
        //print("value aft sampleWeight",(Operand<TFloat32>)values);

        Operand<TFloat32> valueSum = tf.dtypes.cast(tf.reduceSum(values, K.allAxis(tf, values)), TFloat32.DTYPE);
        
        //print("valuesum",valueSum);
        //Op totalUpdate = tf.assignAddVariableOp(total, valueSum);
        Op totalUpdate = tf.assignAdd(total, valueSum);
        updateOperations.add(totalUpdate);
        Operand numValues;
        if (reduction != Reduction.SUM) {
            switch (reduction) {
                case SUM_OVER_BATCH_SIZE:
                    numValues = tf.dtypes.cast(
                            tf.constant(values.asTensor().shape().size()),
                            dtype);
                    break;
                case WEIGHTED_MEAN:
                    if (sampleWeight == null) {
                        numValues = tf.dtypes.cast(tf.constant(values.asOutput().shape().size()), dtype);
                    } else {
                        numValues = tf.reduceSum(sampleWeight, K.allAxis(tf, values));
                    }   break;
                default:
                    throw new UnsupportedOperationException(
                            String.format("reduction [%s] not implemented", reduction));
            }
            Op totalCount = tf.assignAdd(count, tf.dtypes.cast(numValues, TFloat32.DTYPE));
            updateOperations.add(totalCount);
        }
        Scope scope = tf.scope().withSubScope("updateState");
        scope = scope.withControlDependencies(updateOperations);
        return NoOp.create(scope);

    }
    
    @Override
    public Operand result() {
        switch(this.reduction) {
            case SUM:
                return tf.identity(this.total);
            case WEIGHTED_MEAN:
            case SUM_OVER_BATCH_SIZE:
                return tf.math.divNoNan(total.asOutput(), tf.dtypes.cast(count.asOutput(), total.asOutput().dataType()));
            default:
                throw new UnsupportedOperationException(
                            String.format("reduction [%s] not implemented", reduction));
        }
    }
    
    private void print(String prefix, Operand<TFloat32> operand) {
        if(tf.scope().env().isGraph()) {
            try(Session session = new Session((Graph)tf.scope().env())) {
                AtomicInteger index = new AtomicInteger();
                try ( Tensor<TFloat32> result = session.runner().fetch(operand).run().get(0).expect(TFloat32.DTYPE)) {
                    if(result.data().size() > 1) {
                        result.data().scalars().forEach(f -> {
                                System.out.printf("%s: %d). %f\n", prefix, index.incrementAndGet(), f.getFloat());
                        });
                    }else {
                        System.out.printf("%s: %d). %f\n", prefix, index.incrementAndGet(), result.data().getFloat());
                    }
                }
            }
        }
    }

}
