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
public class ZerosTest {
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    private int counter;
    
    public ZerosTest() {
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
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallUInt() {
        System.out.println("call UInt");
        byte[] expected = { 0,0, 0, 0};
        byte[] actual = { 1, 1, 1, 1 }; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Zeros<TUint8> instance = new Zeros<>();
            Operand<TUint8> operand = instance.call(tf, tf.constant(shape.asArray()),  TUint8.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           // operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getI()));
            assertArrayEquals(expected, actual);
        }
    }
    
    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallInt() {
        System.out.println("call Int");
        int[] expected = { 0,0, 0, 0};
        int[] actual = { 1, 1, 1, 1 }; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Zeros<TInt32> instance = new Zeros<>();
            Operand<TInt32> operand = instance.call(tf, tf.constant(shape.asArray()),  TInt32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           // operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getI()));
            assertArrayEquals(expected, actual);
        }
    }
    
    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallLong() {
        System.out.println("call Long");
        long[] expected = { 0,0, 0, 0};
        long[] actual = { 1, 1, 1, 1 }; // init to ones to make sure they all changet to zero
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Zeros<TInt64> instance = new Zeros<>();
            Operand<TInt64> operand = instance.call(tf, tf.constant(shape.asArray()),  TInt64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
           // operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getI()));
            assertArrayEquals(expected, actual);
        }
    }
    

    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallFloat() {
        System.out.println("call float");
        float[] expected = { 0.F,0.F, 0.F, 0.F};
        float[] actual = { 1.f, 1.f, 1.f, 1.f };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
            Zeros<TFloat32> instance = new Zeros<>();
            Operand<TFloat32> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat32.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getFloat()));
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallDouble() {
        System.out.println("call double");
        double[] expected = {  0.,0., 0., 0.};
        double[] actual = { 1., 1., 1., 1. };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
           
            Zeros<TFloat64> instance = new Zeros<>();
            Operand<TFloat64> operand = instance.call(tf, tf.constant(shape.asArray()),  TFloat64.DTYPE);
            operand.asTensor().data().read(DataBuffers.of(actual));
            operand.asTensor().data().scalars().forEach(s -> System.out.println(s.getDouble()));
            assertArrayEquals(expected, actual, EPSILON);
        }
    }
    
    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallString() {
        System.out.println("call String");
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
           
            Zeros<TString> instance = new Zeros<>();
            Operand<TString> operand = instance.call(tf, tf.constant(shape.asArray()),  TString.DTYPE);
            counter = 0;
            operand.asTensor().data().scalars().forEach(s -> {counter++; assertTrue(s.getObject().isEmpty());});
            assertEquals(counter, 2*2);
        }
    }
    
        
    /**
     * Test of call method, of class Zeros.
     */
    @Test
    public void testCallBool() {
        System.out.println("call Boolean");
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            Shape shape = Shape.of(2,2);
           
            Zeros<TBool> instance = new Zeros<>();
            Operand<TBool> operand = instance.call(tf, tf.constant(shape.asArray()),  TBool.DTYPE);
            counter = 0;
            operand.asTensor().data().scalars().forEach(s -> {counter++; assertFalse(s.getBoolean());});
            assertEquals(counter, 2*2);
        }
    }
    
}
