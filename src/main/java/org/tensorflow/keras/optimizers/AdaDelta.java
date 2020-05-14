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

import java.util.HashMap;
import java.util.Map;
import org.tensorflow.Graph;

/**
 * AdaDelta Optimizer that implements the AdaDelta algorithm.
 * Keras wrapper around the Tensorflow Framework optimizer.
 * Adadelta optimization is a stochastic gradient descent method that is based 
 * on adaptive learning rate per dimension to address two drawbacks: 
 * 1) the continual decay of learning rates throughout training 
 * 2) the need for a manually selected global learning rate
 * 
 * Two accumulation steps are required: 1) the accumulation of gradients squared,
 * 2) the accumulation of updates squared.
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
    
    private Map<String, Object> config = new HashMap<>();
    
    // TODO is this still necessary?
    private String[] allowed_options = {"clipnorm", "clipvalue", "lr", "decay"};
    

    /**
     * create an Adadelta optimizer with default name="Adadelta", 
     * learning_rate=0.001F, rho=0.95F, and epsilon=1e-7F
     * @param graph the tensorflow graph
     */
    public AdaDelta(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    /**
     * create an Adadelta optimizer with default learning_rate=0.001F, 
     * rho=0.95F, and epsilon=1e-7F
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adadelta"
     */
    public AdaDelta(Graph graph, String name) {
        this(graph, LEARNING_RATE_DEFAULT, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    
    /**
     * create an Adadelta  optimizer with default name="Adadelta", 
     *  rho=0.95F, and epsilon=1e-7F
     * @param graph  the tensorflow graph
     * @param learningRate The learning rate
     */
    public AdaDelta(Graph graph, float learningRate) {
        this(graph, learningRate, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    /**
     * create an Adadelta  optimizer with default  
     *  rho=0.95F, and epsilon=1e-7F
     * @param graph  the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adadelta"
     * @param learningRate The learning rate
     */
    public AdaDelta(Graph graph, String name, float learningRate) {
        this(graph, learningRate, RHO_DEFAULT, EPSILON_DEFAULT);
    }
    
    
    
    /**
     * create an Adadelta  optimizer with default name="Adadelta", 
     * @param graph  the tensorflow graph
     * @param learningRate The learning rate
     * @param rho The decay rate.
     * @param epsilon A constant epsilon used to better conditioning the grad update.
     */
    public AdaDelta(Graph graph,float learningRate, float rho, float epsilon) {
        super(graph, learningRate, rho, epsilon);
        initConfig(learningRate, rho, epsilon);
    }
    
    /**
     * create an Adadelta  optimizer
     * @param graph  the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adadelta"
     * @param learningRate The learning rate
     * @param rho The decay rate.
     * @param epsilon A constant epsilon used to better conditioning the grad update.
     */
    public AdaDelta(Graph graph, String name, float learningRate, float rho, float epsilon) {
        super(graph, name, learningRate, rho, epsilon);
        initConfig(learningRate, rho, epsilon);
    }
    


     /* TODO - do we need to do this to be compatible with keras python? */
    /**
     * create an Adam Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, he config object has keys for 
     * "name", "learning_rate", "rho" and "epsilon". If a key is missing 
     * the default value is used.
     */
    public static AdaDelta fromConfig(Graph graph, Map<String, Object> config) {
        return create(graph, config);
    }
    
    /**
     * create an Adadelta  optimizer
     *
     * @param graph  the tensorflow graph
     * @@param config a config object to initialize, the config object has keys for 
     * "name", "learning_rate", "rho" and "epsilon". If a key is missing 
     * the default value is used.
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
     * Initialize the configuration based on which constructor is called.
     * @param learningRate The learning rate
     * @param rho The decay rate.
     * @param epsilon  A constant epsilon used to better conditioning the grad update.
     */
    private void initConfig(float learningRate, float rho, float epsilon) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(RHO_RATE_KEY, rho);
        config.put(EPSILON_KEY, epsilon);
    }
    
    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        return config;
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
