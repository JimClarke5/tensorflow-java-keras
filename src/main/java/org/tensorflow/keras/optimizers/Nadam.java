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
import static org.tensorflow.keras.optimizers.OptimizerInterface.NAME_KEY;

/**
 * ********** TODO ******************
 */
/**
 * Nadam Optimizer that implements the NAdam algorithm.
 *
 * @author Jim Clarke
 * @param <U> The Type for the call operation
 */
public class Nadam extends Adam implements OptimizerInterface {

    private Map<String, Object> config = new HashMap<>();

    public Nadam(Graph graph) {
        this(graph, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT);
    }

    public Nadam(Graph graph, String name) {
        this(graph, name, LEARNING_RATE_DEFAULT, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT);
    }

    public Nadam(Graph graph, float learningRate) {
        this(graph, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT);
    }

    public Nadam(Graph graph, String name, float learningRate) {
        this(graph, name, learningRate, BETA_ONE_DEFAULT, BETA_TWO_DEFAULT, EPSILON_DEFAULT);
    }

    public Nadam(Graph graph, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }

    public Nadam(Graph graph, String name, float learningRate, float betaOne, float betaTwo, float epsilon) {
        super(graph, name, learningRate, betaOne, betaTwo, epsilon);
        initConfig(learningRate, betaOne, betaTwo, epsilon);
    }

    /**
     * create an Adam
     *
     * @param graph
     * @param config a config object to initialize
     */
    public static Nadam create(Graph graph, Map<String, Object> config) {
        String name = (String) config.get(NAME_KEY);
        float learningRate = (float) config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float epsilon = (float) config.getOrDefault(EPSILON_KEY, EPSILON_DEFAULT);
        float betaOne = (float) config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        float betaTwo = (float) config.getOrDefault(LEARNING_RATE_KEY, LEARNING_RATE_DEFAULT);
        if (name == null) {
            return new Nadam(graph, learningRate, betaOne, betaTwo, epsilon);
        } else {
            return new Nadam(graph, name, learningRate, betaOne, betaTwo, epsilon);
        }
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Map<String, Object> getConfig() {
        return config;
    }

    /**
     * TODO
     *
     * @Override protected <T extends TType> Op applyDense(Output<T> gradient,
     * Output<T> variable) { Variable<T> firstMomentSlot = getSlot(variable,
     * FIRST_MOMENT).get(); Variable<T> secondMomentSlot = getSlot(variable,
     * SECOND_MOMENT).get(); }
    * *
     */
}
