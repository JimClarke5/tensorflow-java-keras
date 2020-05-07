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
import org.tensorflow.op.Ops;
import org.tensorflow.tools.Shape;
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;

/**
 *
 * @author Jim Clarke
 */
public class GlorotNormalTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;
    
    public GlorotNormalTest() {
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
     * Test of getConfig method, of class GlorotNormal.
     */
    @Test
    public void testGetConfig() {
        System.out.println("getConfig");
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotNormal.SCALE_KEY, 1.0);
        config.put(GlorotNormal.MODE_KEY, "fan_avg");
        config.put(GlorotNormal.DISTRIBUTION_KEY, "truncated_normal");
        config.put(GlorotNormal.SEED_KEY, SEED);    
        GlorotNormal instance = new GlorotNormal(SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class GlorotNormal.
     */
    @Test
    public void testConfigCTORMap() {
        System.out.println("ctor Map");
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotNormal.SCALE_KEY, 1.0);
        config.put(GlorotNormal.MODE_KEY, "fan_avg");
        config.put(GlorotNormal.DISTRIBUTION_KEY, "truncated_normal");
        config.put(GlorotNormal.SEED_KEY, SEED);    
        GlorotNormal instance = new GlorotNormal(config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class GlorotNormal.
     */
    @Test
    public void testCall_Float() {
        System.out.println("call Float");
        float[] actual = { 0,0, 0, 0};
        float[] expected = { -0.21565685F, 1.2418709F, 0.1305355F, 1.0191818F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            GlorotNormal<TFloat32> instance = 
                    new GlorotNormal<>(SEED);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getFloat()));
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double() {
        System.out.println("call Double");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 1.1804090085115349, 0.028932303309678203, 
                                -0.5210409914947474, -1.1755309088162063 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            GlorotNormal<TFloat64> instance = 
                    new GlorotNormal<>(SEED);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getDouble()));
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
}
