/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.backend;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Variable;
import org.tensorflow.op.math.Mean;
import org.tensorflow.tools.Shape;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;

/**
 *
 * @author Jim Clarke
 */
public class K {
    public static final double Epsilon = 1e-7;
    public static final float EpsilonF = 1e-7F;
    
    public static final double epsilon() { return Epsilon; }
    
    public static final Operand epsilonConstant(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(Epsilon), dtype); }
    public static final Operand one(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(1), dtype); }
    public static final Operand zero(Ops tf, DataType dtype) { return tf.dtypes.cast(tf.constant(1), dtype); }
    public static final Operand constant(Ops tf, double number, DataType dtype) { return tf.dtypes.cast(tf.constant(number), dtype); }
    
    
    public static Operand clip(Ops tf, Operand x, double minValue, double maxValue) {
        assert x != null : "Operand x must not be null";
        DataType dtype = x.asOutput().dataType();
        if(maxValue < minValue) {
            maxValue = minValue;
        }
        Operand minValueConstant = tf.dtypes.cast(tf.constant(minValue), dtype);
        Operand maxValueConstant = tf.dtypes.cast(tf.constant(maxValue), dtype);
        return tf.clipByValue(x, minValueConstant, maxValueConstant);
    }
    
    public static Operand mean(Ops tf, Operand x) {
        return mean(tf, x, tf.constant(-1), false);
    }
    public static Operand mean(Ops tf, Operand x, Operand axis) {
        return mean(tf, x, axis, false);
    }
    
    public static Operand mean(Ops tf, Operand x,boolean keepDims) {
        return mean(tf, x, tf.constant(-1), keepDims);
    }
    
    public static Operand mean(Ops tf, Operand x, Operand axis, boolean keepDims) {
        if(x.asOutput().dataType() == TBool.DTYPE) {
            x = tf.dtypes.cast(x, TFloat32.DTYPE);
        }
        return tf.math.mean(x, axis, Mean.keepDims(keepDims));
    }
    
    public static Operand maximum(Ops tf, Operand x, Operand y) {
        y = tf.dtypes.cast(y, x.asOutput().dataType());
        return tf.math.maximum(x,y);
    }
    
    public static Shape merge(Shape a, Shape b) {
        assert a.numDimensions() == b.numDimensions() : String.format("Shapes %s and %s are incompatible", a, b);
        long[] array = new long[a.numDimensions()];
        for(int i = 0; i < a.numDimensions(); i++) {
            if(a.size(i) != Shape.UNKNOWN_SIZE) {
                if(b.size(i) != Shape.UNKNOWN_SIZE)
                    assert a.size(i) == b.size(i) : String.format("Shapes %s and %s are incompatible", a, b);
                array[i] = a.size(i);
            } else
                array[i] = b.size(i);
        }
        return Shape.of(array);
        
    }
    
    
    // this is from nn in Python, I could not find it in the Java frameworks.
    public static Operand sigmoidCrossEntropyWithLogits(Ops tf, Operand labels, Operand logits) {
        Shape lablesShape = labels.asOutput().shape();
        Shape logitsShape = logits.asOutput().shape();
        Shape newShape = merge(lablesShape, logitsShape);
        
        Operand zeros = tf.dtypes.cast(tf.zerosLike(logits), logits.asOutput().dataType());
        Operand cond = tf.math.greaterEqual(logits, zeros);
        
        Operand relu_logits = tf.select(cond, logits, zeros);
        Operand neg_abs_logits = tf.select(cond, tf.math.neg(logits), logits);
        return tf.math.add(
             tf.math.sub(relu_logits, tf.math.mul(logits, labels)),
             tf.math.log1p(tf.math.exp(neg_abs_logits))
        );
        
    }
    // Not sure if this is right.
    private static Operand  backtrackIdentity(Operand output) {
        while(!output.op().type().equals("Identity"))
            output = output.op().output(0);
        return output;
    }
    
    public static Operand binary_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits, Session session ){
        if(fromLogits) {
            return sigmoidCrossEntropyWithLogits(tf, target, output);
        }
        
        if(!(output instanceof Variable) && (!tf.scope().env().isEager())) {
            //output = backtrackIdentity(output); // TODO - this dose not work, goes infinite loop
            if(output.op().type().equals("Sigmoid")) {
                assert output.op().numOutputs() == 1;
                output = output.op().output(0);
                return sigmoidCrossEntropyWithLogits(tf, target, output);
            }
        }
        DataType dtype = output.asOutput().dataType();
        Operand one = one(tf,dtype);
        Operand epsilonConst = K.epsilonConstant(tf,dtype);
        Operand oneMinusEpsilonConst = tf.math.sub(one, epsilonConst);
        output = tf.clipByValue(output, epsilonConst, oneMinusEpsilonConst);
        debug( session, "BCE/output",output);
        
        // Compute cross entropy from probabilities.
        Operand bce = tf.math.mul(target, tf.math.log(tf.math.add(output, epsilonConst)));
        debug( session, "BCE/bce1",bce);
        bce = tf.math.add(bce,
            tf.math.mul(
                tf.math.sub(one, target),
                tf.math.log(tf.math.add(tf.math.sub(one, output), epsilonConst )) 
            ));
        debug( session, "BCE/bce2",bce);
        Operand result =  tf.math.neg(bce);
        debug( session, "BCE/result",result);
        return result;
    }
    
    public static Op categorical_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits) {
        return categorical_crossentropy(tf, target, output, fromLogits, -1);
    }

    public static Op categorical_crossentropy(Ops tf, Operand target, Operand output, boolean fromLogits, int axis) {
        if(fromLogits) {
            return tf.nn.softmaxCrossEntropyWithLogits(target, output);
        }
        if(!(output instanceof Variable) && (!tf.scope().env().isEager())) {
            output = backtrackIdentity(output);
            if(output.op().type().equals("Softmax")) {
                assert output.op().numOutputs() == 1;
                output = output.op().output(0);
                return  tf.nn.softmaxCrossEntropyWithLogits( target, output);
            }
        }
        DataType dtype = output.asOutput().dataType();
        Operand one = one(tf,dtype);
        Operand epsilonConst = K.epsilonConstant(tf,dtype);
        Operand oneMinusepsilonConst = tf.math.sub(one, epsilonConst);
        output = tf.clipByValue(output, epsilonConst, oneMinusepsilonConst);
        
        // Compute cross entropy from probabilities.
        Operand bce = tf.math.mul(target, tf.math.log(tf.math.add(output, epsilonConst)));
        bce = tf.math.add(bce,
            tf.math.mul(
                tf.math.sub(one(tf, dtype), target),
                tf.math.log(tf.math.sub(one, tf.math.add(output, epsilonConst)))
            ));
        return tf.math.neg(bce);
    }
    
    private static int[] allAxis(Operand op) {
        int rank = op.asOutput().shape().numDimensions();
        int[] ranks = new int[rank];
        for(int i = 0; i < rank; i++)
            ranks[i] = i;
        return ranks;
    }
    
    public static Operand allAxis(Ops tf, Operand op) {
        int[] ranks = allAxis(op);
        return tf.constant(ranks);
    }
    
    private static void debug(Session session, String prefix, Operand operand) {
        if(session != null) {
            try ( Tensor<TFloat32> result = session.runner().fetch(operand).run().get(0).expect(TFloat32.DTYPE)) {
                        result.data().scalars().forEach(f -> {
                            System.out.printf("%s:  Actual = %f\n", prefix, f.getFloat());
                         });
            }
        }
    }
    
}
