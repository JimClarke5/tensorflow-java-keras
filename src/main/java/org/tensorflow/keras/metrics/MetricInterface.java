/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.metrics;

import org.tensorflow.Operand;
import org.tensorflow.op.Op;

/**
 *
 * @author Jim Clarke
 */
public interface MetricInterface {
    
    /**
     * reset states 
     */
    public Op resetStates();
    
  
    
    /**
     * update States
     * 
     * @param args Operands
     * @return the updated State
     */
    public Op updateState(Operand... args);
    
    /**
     * get the result of the metric
     * 
     * @return the result;
     */
    public Operand result();
}
