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
public class VarianceScalingTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    private static final long SEED = 1000L;
    
    public VarianceScalingTest() {
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
     * Test of getConfig method, of class VarianceScaling.
     */
    @Test
    public void testGetConfig() {
        System.out.println("getConfig");
        Map<String, Object> config = new HashMap<>();
        config.put(VarianceScaling.SCALE_KEY, 1.0);
        config.put(VarianceScaling.MODE_KEY, "fan_in");
        config.put(VarianceScaling.DISTRIBUTION_KEY, "truncated_normal");
        config.put(VarianceScaling.SEED_KEY, SEED);    
        VarianceScaling instance = new VarianceScaling(SEED);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class VarianceScaling.
     */
    @Test
    public void testConfigCTORMap() {
        System.out.println("ctor Map");
        Map<String, Object> config = new HashMap<>();
        config.put(VarianceScaling.SCALE_KEY, 1.0);
        config.put(VarianceScaling.MODE_KEY, "fan_in");
        config.put(VarianceScaling.DISTRIBUTION_KEY, "truncated_normal");
        config.put(VarianceScaling.SEED_KEY, SEED);    
        VarianceScaling instance = new VarianceScaling(config);
        Map<String, Object> expResult = config;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

    /**
     * Test of call method, of class VarianceScaling.
     */
    @Test
    public void testCall_Float_1_FAN_IN_TRUNCATED_NORMAL() {
        System.out.println("call Float_1_FAN_IN_TRUNCATED_NORMAL");
        float[] actual = { 0,0, 0, 0};
        float[] expected = { -0.52388954F, -0.29329166F, -0.07872587F, -0.31851602F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat32> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "truncated_normal", SEED);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double_1_FAN_IN_TRUNCATED_NORMAL() {
        System.out.println("call Double_1_FAN_IN_TRUNCATED_NORMAL");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 1.4971264721246893, -1.2488522307109322, 
                            -0.5409677352523339, 0.4871390504288623 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat64> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "truncated_normal", SEED);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
    /**
     * Test of call method, of class VarianceScaling.
     */
    @Test
    public void testCall_Float_1_FAN_IN_UNTRUNCATED_NORMAL() {
        System.out.println("call Float_1_FAN_IN_UNTRUNCATED_NORMAL");
        float[] actual = { 0,0, 0, 0};
        float[] expected = { -0.46082667F, -0.25798687F, -0.06924929F, -0.28017485F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat32> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "untruncated_normal", SEED);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double_1_FAN_IN_UNTRUNCATED_NORMAL() {
        System.out.println("call Double_1_FAN_IN_UNTRUNCATED_NORMAL");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 1.3169108626945392, -1.0985224689731887, 
                            -0.13536536217837225, -1.698770780615686 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat64> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "untruncated_normal", SEED);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
     /**
     * Test of call method, of class VarianceScaling.
     */
    @Test
    public void testCall_Float_1_FAN_IN_UNIFORM() {
        System.out.println("call Float_1_FAN_IN_UNIFORM");
        float[] actual = { 0,0, 0, 0};
        float[] expected = { 0.9266439F, 0.8190767F, 1.1268647F, 0.6596042F };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat32> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "uniform", SEED);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON_F);
        }
    }
    
    @Test
    public void testCall_Double_1_FAN_IN_UNIFORM() {
        System.out.println("call Double_1_FAN_IN_UNIFORM");
        double[] actual = { 0,0, 0, 0};
        double[] expected = { 0.06468193804916589, 0.44170328686673477, 
                        0.06711059208157763, 0.6278720842445181 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            VarianceScaling<TFloat64> instance = 
                    new VarianceScaling<>(1.0, "fan_in", "uniform", SEED);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(actual,expected, EPSILON);
        }
    }
    
}
