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
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;
import static org.tensorflow.keras.optimizers.OptimizerInterface.config;

/**
 * Adam Optimizer that implements the Adam algorithm.
 *
 * @author Jim Clarke
 */
public class Adam extends org.tensorflow.framework.optimizers.Adam implements OptimizerInterface  {
    public static final String LEARNING_RATE_KEY = "learning_rate";
    public static final String EPSILON_KEY = "epsilon";
    public static final String BETA_ONE_KEY = "beta_1";
    public static final String BETA_TWO_KEY = "beta_2";
    
    public static final float  LEARNING_RATE_DEFAULT = 0.001F;
    public static final float  EPSILON_DEFAULT = 1e-07F;
    public static final float  BETA_ONE_DEFAULT = 0.9F;
    public static final float  BETA_TWO_DEFAULT = 0.999F;
    

    /**
     * create an Adam Optimizer
     * @param graph the tensorflow graph
     */
    public Adam(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    
    /**
     * create an Adam Optimizer
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adam"
     */
    public Adam(Graph graph, String name) {
        this(graph, name, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    
    /**
     * create an Adam Optimizer
     * @param graph the tensorflow graph
     * @param learningRate The learning rate. Defaults to 0.001.
     */
    public Adam(Graph graph, float learningRate) {
        this(graph, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    
    /**
     * create an Adam Optimizer
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adam"
     * @param learningRate The learning rate. Defaults to 0.001.
     */
    public Adam(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT );
    }
    /**
     *  create an Adam Optimizer
     * 
     * @param graph the tensorflow graph
     * @param learningRate The learning rate. Defaults to 0.001.
     * @param betaOne The exponential decay rate for the 1st moment estimates. 
     * Defaults to 0.9.
     * @param betaTwo The exponential decay rate for the 2nd moment estimates. 
     * Defaults to 0.999.
     * @param epsilon A small constant for numerical stability. This epsilon is
     * "epsilon hat" in the Kingma and Ba paper (in the formula just before
     *  Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
     */
    public Adam(Graph graph, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }
    
    /**
     *  create an Adam Optimizer
     * 
     * @param graph the tensorflow graph
     * @param name the name of the Optimizer, defaults to "Adam"
     * @param learningRate The learning rate. Defaults to 0.001.
     * @param betaOne The exponential decay rate for the 1st moment estimates. 
     * Defaults to 0.9.
     * @param betaTwo The exponential decay rate for the 2nd moment estimates. 
     * Defaults to 0.999.
     * @param epsilon A small constant for numerical stability. This epsilon is
     * "epsilon hat" in the Kingma and Ba paper (in the formula just before
     *  Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
     */
    public Adam(Graph graph, String name, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, name, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }
    
    /* TODO - do we need to do this to be compatible with keras python? */
    /**
     * create an Adam Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, the config object has keys for 
     * "name", "learning_rate", "epsilon", "beta_1", "beta_2". If a key is missing 
     * the default value is used.
     */
    public static Adam fromConfig(Graph graph, Map<String, Object> config) {
        return create(graph, config);
    }
    

    /**
     * create an Adam Optimizer from a config object
     *
     * @param graph the tensorflow graph
     * @param config a config object to initialize, the config object has keys for 
     * "name", "learning_rate", "epsilon", "beta_1", "beta_2". If a key is missing 
     * the default value is used.
     */
    public static  Adam create(Graph graph, Map<String, Object> config) {
        String name = (String)config.get(NAME_KEY);
        float learningRate = (float)config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float epsilon = (float)config.getOrDefault(EPSILON_KEY, EPSILON_DEFAULT);
        float betaOne = (float)config.getOrDefault(BETA_ONE_KEY, BETA_ONE_DEFAULT);
        float betaTwo = (float)config.getOrDefault(BETA_TWO_KEY, BETA_TWO_DEFAULT);
        if(name == null) 
            return new Adam(graph, learningRate, betaOne,betaTwo, epsilon );
        else
            return new Adam(graph, name, learningRate, betaOne,betaTwo, epsilon );
    }
    
    /**
     * function that sets the config object based on which constructor is called.
     * 
     * @param learningRate The learning rate. Defaults to 0.001.
     * @param betaOne The exponential decay rate for the 1st moment estimates. 
     * Defaults to 0.9.
     * @param betaTwo The exponential decay rate for the 2nd moment estimates. 
     * Defaults to 0.999.
     * @param epsilon A small constant for numerical stability. This epsilon is
     * "epsilon hat" in the Kingma and Ba paper (in the formula just before
     *  Section 2.1), not the epsilon in Algorithm 1 of the paper. Defaults to 1e-7.
     */
    protected void initConfig(float learningRate, float betaOne, float betaTwo, float epsilon) {
        config.put(NAME_KEY, this.getOptimizerName());
        config.put(LEARNING_RATE_KEY, learningRate);
        config.put(EPSILON_KEY, epsilon);
        config.put(BETA_ONE_KEY, betaOne);
        config.put(BETA_TWO_KEY, betaTwo);
        
    }
}
