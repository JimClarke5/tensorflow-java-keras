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
public class GlorotUniformTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;
    
    int counter;
    
    public GlorotUniformTest() {
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
     * Test of getConfig method, of class GlorotUniform.
     */
    @Test
    public void testGetConfig() {
        System.out.println("getConfig");
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotUniform.SCALE_KEY, 1.0);
        config.put(GlorotUniform.MODE_KEY, "fan_avg");
        config.put(GlorotUniform.DISTRIBUTION_KEY, "uniform");
        config.put(GlorotUniform.SEED_KEY, SEED);    
        GlorotUniform instance = new GlorotUniform(null, SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class GlorotUniform.
     */
    @Test
    public void testConfigCTORMap() {
        System.out.println("ctor Map");
        Map<String, Object> config = new HashMap<>();
        config.put(GlorotUniform.SCALE_KEY, 1.0);
        config.put(GlorotUniform.MODE_KEY, "fan_avg");
        config.put(GlorotUniform.DISTRIBUTION_KEY, "uniform");
        config.put(GlorotUniform.SEED_KEY, SEED);    
        GlorotUniform instance = new GlorotUniform(null, config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class GlorotUniform.
     */
    @Test
    public void testCall_Float() {
        System.out.println("call Float");
        float[] actual = { 0,0, 0, 0};
        float[] expected = {0.9266439F, 0.8190767F, 1.1268647F, 0.6596042F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            GlorotUniform<TFloat32> instance = 
                    new GlorotUniform<>(tf, SEED);
            Operand<TFloat32> operand = instance.call(tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            //ounter = 0;
            //operand.asTensor().data().scalars().forEach(s -> {counter++;});
            //assertEquals(counter, 2*2);
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double() {
        System.out.println("call Double");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 0.06468193804916589, 0.44170328686673477,
            0.06711059208157763, 0.6278720842445181};
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            GlorotUniform<TFloat64> instance = 
                    new GlorotUniform<>(tf, SEED);
            Operand<TFloat64> operand = instance.call(tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            //counter = 0;
            //operand.asTensor().data().scalars().forEach(s -> {counter++;});
            //assertEquals(shape.size(), counter);
            
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
    
}
