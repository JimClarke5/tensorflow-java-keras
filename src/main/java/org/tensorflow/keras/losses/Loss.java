/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

import java.util.HashMap;
import java.util.Map;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.keras.losses.impl.LossesImpl;
import org.tensorflow.op.Ops;
import org.tensorflow.types.family.TNumber;

/**
 * Base class for Loss
 *
 * @author Jim Clarke
 */
public abstract class Loss implements LossFunction {

    public static final double EPSILON = 1e-7;
    public static final float EPSILON_F = 1e-7F;

    private final Reduction reduction;
    private final String name;
    private final Map<String, Object> config = new HashMap<>();
    
    protected final Ops tf;
    
    // for debug
    private Session session;

    /**
     * create a loss with  name = class name and reduction = AUTO
     */
    protected Loss(Ops tf) {
        this(tf, null, Reduction.AUTO);
    }

    /**
     * create a loss with reduction = AUTO 
     *
     * @param name the name of the Loss Function
     */
    protected Loss(Ops tf, String name) {
        this(tf, name, Reduction.AUTO);
    }


    /**
     * create a loss 
     *
     * @param name the name of the Loss Function
     * @param reduction the reduction
     */
    protected Loss(Ops tf, String name, Reduction reduction) {
        
        this.name = name == null ? this.getClass().getSimpleName() : name;
        this.reduction = reduction;
        this.tf = tf != null ? tf.withSubScope(this.name) : null;
    }


    /**
     * Calculates the loss
     * 
     * @param <T> Operands extend TNumber
     * @param tf the TensorFlow Ops
     * @param labels the truth values or labels
     * @param predictions the predictions
     * @return the loss
     */
    public <T extends TNumber> Operand<T> call(Operand<T> labels, Operand<T> predictions) {
        return call(labels, predictions, null);
    }

    protected <T extends TNumber> Operand computeWeightedLoss(Operand<T> losses, Reduction reduction, Operand sampleWeight) {
        return LossesImpl.computeWeightedLoss(tf, losses, reduction, sampleWeight);
    }

    /**
     * @return the reduction
     */
    public Reduction getReduction() {
        return reduction;
    }

    /**
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * @return the config
     */
    public Map<String, Object> getConfig() {
        return config;
    }
    
    public boolean isDebug() {
        return this.session != null;
    }

    /**
     * @return the session
     */
    public Session getSession() {
        return session;
    }

    /**
     * @param session the session to set
     */
    public void setDebug(Session session) {
        this.session = session;
        Losses.setDebug(session);
    }

    /**
     * @return the tf
     */
    public Ops getTf() {
        return tf;
    }



}
