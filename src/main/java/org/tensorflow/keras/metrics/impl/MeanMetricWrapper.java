/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics.impl;

import org.tensorflow.DataType;
import org.tensorflow.Operand;
import org.tensorflow.keras.losses.LossFunction;
import org.tensorflow.keras.metrics.Mean;
import org.tensorflow.op.Op;
import org.tensorflow.op.Ops;

/**
 *
 * @author Jim Clarke
 */
public class MeanMetricWrapper extends Mean {
    protected  LossFunction loss;
    
    public MeanMetricWrapper(Ops tf) {
        this(tf, null, null);
    }
    
    public MeanMetricWrapper(Ops tf, String name ) {
        this(tf, name, null);
    }
    
    public MeanMetricWrapper(Ops tf, DataType dType) {
        this(tf, null, dType);
    }
    
    public MeanMetricWrapper(Ops tf, String name, DataType dType) {
        super(tf, name, dType);
    }
    
    
    public final void setLoss(LossFunction loss) {
        this.loss = loss;
    }

    @Override
    public Op updateState(Operand... operands) {
        Operand labels = operands[0];
        Operand predictions = operands[1];
        Operand sampleWeights = operands.length > 2 ? operands[2] : null;
        Operand losses = loss.call(labels, predictions, null);
        return super.updateState(losses, sampleWeights);        
    }
    
    
}
