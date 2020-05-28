/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.util.function.Supplier;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TBool;
/**
 *
 * @author Jim Clarke
 */
public class SmartCond {
    
    public static Operand cond(Boolean pred, Supplier<Operand> true_fn, Supplier<Operand> false_fn){
        assert pred != null : "pred must not be null";
        assert true_fn != null : "true_fn must not be null";
        assert false_fn != null : "false_fn must not be null";
        return pred ? true_fn.get() : false_fn.get();
    }
    public static Operand cond(Number pred, Supplier<Operand> true_fn, Supplier<Operand> false_fn){
        assert pred != null : "pred must not be null";
        assert true_fn != null : "true_fn must not be null";
        assert false_fn != null : "false_fn must not be null";
        return pred.intValue() == 1 ? true_fn.get() : false_fn.get();
    }
    
    public static Operand cond(String pred, Supplier<Operand> true_fn, Supplier<Operand> false_fn){
        assert pred != null : "pred must not be null";
        assert true_fn != null : "true_fn must not be null";
        assert false_fn != null : "false_fn must not be null";
        return Boolean.valueOf(pred) ? true_fn.get() : false_fn.get();
    }
    
    // TODO Select doesn't take a lambda, so what is benefit of using one?
    public static Operand cond(Ops tf, Operand<TBool> pred, Supplier<Operand> true_fn, Supplier<Operand> false_fn){
        assert pred != null : "pred must not be null";
        assert true_fn != null : "true_fn must not be null";
        assert false_fn != null : "false_fn must not be null";
        return tf.select(pred, true_fn.get(), false_fn.get());
    }
    
    public static Operand cond(Ops tf, Operand<TBool> pred, Operand true_op, Operand false_op){
        assert pred != null : "pred must not be null";
        assert true_op != null : "true_op must not be null";
        assert false_op != null : "false_op must not be null";
        return tf.select(pred, true_op, false_op);
    }
    
}
