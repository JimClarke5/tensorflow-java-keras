/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
=======================================================================*/
package org.tensorflow.keras.initializers;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.tensorflow.EagerSession;
import org.tensorflow.Operand;
import org.tensorflow.keras.utils.PrintUtils;
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 *
 * @author Jim Clarke
 */
public class HeUniformTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;

    int counter;

    public HeUniformTest() {
    }

    @BeforeAll
    public static void setUpClass() {
    }

    @AfterAll
    public static void tearDownClass() {
    }

    @BeforeEach
    public void setUp() {
    }

    @AfterEach
    public void tearDown() {
    }

    /**
     * Test of getConfig method, of class HeUniform.
     */
    @Test
    public void testGetConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put(HeUniform.SCALE_KEY, 2.0);
        config.put(HeUniform.MODE_KEY, "fan_in");
        config.put(HeUniform.DISTRIBUTION_KEY, "uniform");
        config.put(HeUniform.SEED_KEY, SEED);
        HeUniform instance = new HeUniform(null, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of getConfig method, of class HeUniform.
     */
    @Test
    public void testConfigCTORMap() {
        Map<String, Object> config = new HashMap<>();
        config.put(HeUniform.SCALE_KEY, 2.0);
        config.put(HeUniform.MODE_KEY, "fan_in");
        config.put(HeUniform.DISTRIBUTION_KEY, "uniform");
        config.put(HeUniform.SEED_KEY, SEED);
        HeUniform instance = new HeUniform(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class HeUniform.
     */
    @Test
    public void testCall_Float() {
        float[] actual = {0, 0, 0, 0};
        float[] expected = {1.3104724F, 1.1583493F, 1.5936272F, 0.93282115F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            HeUniform<TFloat32> instance
                    = new HeUniform<>(tf, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape.asArray()), TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            //counter = 0;
            // operand.asTensor().data().scalars().forEach(s -> {counter++;});
            //assertEquals(counter, 2*2);
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    @Test
    public void testCall_Double() {
        double[] actual = {0, 0, 0, 0};
        double[] expected = {.09147407402970674, 0.6246627788317102,
            0.09490870950065552, 0.8879452169740599};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Shape shape = Shape.of(2, 2);
            HeUniform<TFloat64> instance
                    = new HeUniform<>(tf, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape.asArray()), TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            //ounter = 0;
            //operand.asTensor().data().scalars().forEach(s -> {counter++;});
            //assertEquals(shape.size(), counter);
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

}
