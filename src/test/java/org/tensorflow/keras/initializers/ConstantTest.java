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
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

/**
 *
 * @author Jim Clarke
 */
public class ConstantTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    private static final Map<String, Object> CONFIG_MAP = new HashMap<>();
    static {
        CONFIG_MAP.put("value", 2.0);
        CONFIG_MAP.put("bvalue", null);
    }
    
    private int counter = 0;
    
    public ConstantTest() {
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
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testGetConfigNumber() {
        System.out.println("getConfig");
        Constant instance = new Constant(2.0);
        Map<String, Object> expResult = CONFIG_MAP;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    /**
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testGetConfigBoolean() {
        System.out.println("getConfig");
        Map<String, Object> map = new HashMap<>();
        map.put("value", null);
        map.put("bvalue", true);
        Constant instance = new Constant(true);
        Map<String, Object> expResult = map;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }
    
    /**
     * Test of getConfig method, of class Constant.
     */
    @Test
    public void testGetConfigMapCTOR() {
        System.out.println("ctor Map");
        Constant instance = new Constant(CONFIG_MAP);
        Map<String, Object> expResult = CONFIG_MAP;
        Map<String, Object> result = instance.getConfig();
        assertEquals(expResult, result);
    }

       /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallUInt() {
        System.out.println("call UInt");
        byte[] actual = { 0,0, 0, 0};
        byte[] expected = { 0xf, 0xf, 0xf, 0xf }; // init to constant to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Constant<TUint8> instance = new Constant<>(0xf);
            Operand<TUint8> operand = instance.call(tf, tf.constant(shape.asArray()),  TUint8.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           //PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }
    
    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallInt() {
        System.out.println("call Int");
        int[] actual = { 0,0, 0, 0};
        int[]expected = {0xf, 0xf, 0xf, 0xf }; // init to constant to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Constant<TInt32> instance = new Constant<>(0xf);
            Operand<TInt32> operand = instance.call(tf, tf.constant(shape.asArray()),  TInt32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           // PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }
    
    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallLong() {
        System.out.println("call Long");
        long[] actual = { 0,0, 0, 0};
        long[]expected = { 0xff, 0xff, 0xff, 0xff }; // init to constant to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Constant<TInt64> instance = new Constant<>(0xff);
            Operand<TInt64> operand = instance.call(tf, tf.constant(shape.asArray()),  TInt64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           // PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }
    

    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallFloat() {
        System.out.println("call float");
        float[] actual = { 0.F,0.F, 0.F, 0.F};
        float[]expected = { 12.f, 12.f, 12.f, 12.f };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Constant<TFloat32> instance = new Constant<>(12.F);
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            //PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallDouble() {
        System.out.println("call double");
        double[] actual = {  0.,0., 0., 0.};
        double[]expected = { 11., 11., 11., 11. };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
           
            Constant<TFloat64> instance = new Constant<>(11.);
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            //PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallString() {
        System.out.println("call String");
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
           
            Constant<TString> instance = new Constant<>(22);
            Operand<TString> operand = instance.call(tf, tf.constant(shape.asArray()),  TString.DTYPE);
            fail("AssertionError should have been thrown for TString");
        }catch(AssertionError expected) {
        }
    }
    
        
    /**
     * Test of call method, of class Constant.
     */
    @Test
    public void testCallBool() {
        System.out.println("call Boolean");
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            boolean[] actual = {  false, false, false, false};
            boolean[] expected = { true, true, true, true };
           
            Constant<TBool> instance = new Constant<>(true);
            Operand<TBool> operand = instance.call(tf, tf.constant(shape.asArray()),  TBool.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            counter = 0;
            operand.asTensor().data().scalars().forEach(s -> {counter++;});
            assertEquals(shape.size(), counter);
            // TODO assertArrayEquals(expected, actual);
        }
    }
    
}
