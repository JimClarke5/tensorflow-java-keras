/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the );
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an  BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.optimizers;

import java.util.Map;
import org.tensorflow.Graph;

/**
 * AdaDelta Optimizer that implements the AdaDelta algorithm.
 * Keras wrapper around the Tensorflow Framework optimizer
 *
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */
public class AdaDelta extends org.tensorflow.framework.optimizers.AdaDelta implements OptimizerInterface {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String RHO_RATE_KEY = "rho";
    public static final String EPSILON_KEY = "epsilon";
    
    public static final float  LEARNING_RATE_DEFAULT = 0.001F;
    public static final float  RHO_DEFAULT = 0.95F;
    public static final float  EPSILON_DEFAULT = 1e-7F;
    
    
    // TODO is this still necessary?
    private String[] allowed_options = {"clipnorm", "clipvalue", "lr", "decay"};
    

    /**
     * create an Adadelta
     * @param graph
     */
    public AdaDelta(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    /**
     * create an Adadelta
     * @param graph
     */
    public AdaDelta(Graph graph, String name) {
        this(graph, LEARNING_RATE_DEFAULT, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    
    /**
     * create an Adadelta
     * @param graph
     * @param learningRate
     */
    public AdaDelta(Graph graph, float learningRate) {
        this(graph, learningRate, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    /**
     * create an Adadelta
     * @param graph
     * @param learningRate
     */
    public AdaDelta(Graph graph, String name, float learningRate) {
        this(graph, learningRate, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    
    
    /**
     * create an Adadelta
     * @param graph
     * @param learningRate
     * @param rho
     * @param epsilon
     */
    public AdaDelta(Graph graph,float learningRate, float rho, float epsilon) {
        super(graph, learningRate, rho, epsilon);
        initConfig(learningRate, rho, epsilon);
    }
    
    /**
     * create an Adadelta
     * @param graph
     * @param learningRate
     * @param rho
     * @param epsilon
     */
    public AdaDelta(Graph graph, String name, float learningRate, float rho, float epsilon) {
        super(graph, name, learningRate, rho, epsilon);
        initConfig(learningRate, rho, epsilon);
    }
    


    /**
     * create an Adadelta
     *
     * @param graph
     * @param config a config object to initialize
     */
    public static AdaDelta create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float rho = (float)config.getOrDefault(RHO_RATE_KEY, RHO_DEFAULT);
        float epsilon = (float)config.getOrDefault(EPSILON_KEY, EPSILON_DEFAULT);
        if(name == null)  // doe this to get the default name
            return new AdaDelta(graph, learningRate, rho, epsilon);
        else
            return new AdaDelta(graph, name, learningRate, rho, epsilon);
    }
    
    /**
     * Initialize the configuration
     * @param learningRate
     * @param rho
     * @param epsilon 
     */
    private void initConfig(float learningRate, float rho, float epsilon) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(RHO_RATE_KEY, rho);
        config.put(EPSILON_KEY, epsilon);
    }
    
    
    
    //TODO ??
    //variables()
    //set_weights
    //get_weights
    //add_weight
    //get_updates
    //get_slot_names
    //get_gradients
    //add_slot => createSlot
   


}
