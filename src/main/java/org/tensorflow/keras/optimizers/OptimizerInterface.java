/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.tensorflow.keras.optimizers;

import java.util.HashMap;
import java.util.Map;

/**
 *
 * @author Jim Clarke
 */
public interface OptimizerInterface {
    public static final String NAME_KEY = "name";
    
    Map<String, Object> config = new HashMap<>();
    
    /**
     * @return the config object used to initialize the Initializer values
     */
    default Map<String, Object> getConfig() {
        
        return config;
    }
}
