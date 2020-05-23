/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.losses;

/**
 * Types of loss reduction.
 * 
 * @author Jim Clarke
 */
public enum Reduction {
    AUTO,  NONE, SUM , SUM_OVER_BATCH_SIZE;
    
    /**
     * Get the Reduction based on name
     * @param name the name of the reduction
     * @return the Reduction 
     */
    public static Reduction ofName(String name) {
        return Reduction.valueOf(name.toUpperCase());
    }
}
