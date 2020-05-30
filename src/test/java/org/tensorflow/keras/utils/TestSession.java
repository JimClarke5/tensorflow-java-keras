/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.utils;

import java.io.Closeable;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 *
 * @author Jim Clarke
 */
public abstract class TestSession implements Closeable {
    protected float epsilon = 1e-5F;
    
    public enum Mode { EAGER, GRAPH; }
            
    public static TestSession createEagerSession() {
        return new EagerTestSession();
    }
    
    public static TestSession createGraphSession() {
        return new GraphTestSession();
    }
    
    public static TestSession createTestSession(Mode mode) {
        return mode == Mode.EAGER?  createEagerSession() : createGraphSession();
    }
    
    public <T extends TNumber> void evaluate(Number expected, Operand<T> input) {
        evaluate(new Number[]{ expected } , input);
    }
    
    public abstract <T extends TNumber>void evaluate(Number[] expected, Operand<T> input);
    
    
    public abstract Ops getTF();
    
    public abstract boolean isEager();
    
    public  boolean isGraph() {
        return !isEager();
    }
    
    public float getEpsilon() {
        return this.epsilon;
    }
    
    public void setEpsilon() {
        this.epsilon = epsilon;
    }
    
    public abstract Session getGraphSession();
    public abstract EagerSession getEagerSession();
    
    @Override
    public abstract void close();
    
    
    
    
}
