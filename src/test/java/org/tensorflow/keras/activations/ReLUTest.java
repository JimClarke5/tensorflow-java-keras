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
package org.tensorflow.keras.activations;

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
import org.tensorflow.tools.buffer.DataBuffers;
import org.tensorflow.types.TFloat16;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;

/**
 *
 * @author Jim Clarke
 */
public class ReLUTest {
    
    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;
    
    public ReLUTest() {
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
     * Test of ReLU call method
     */
    @Test
    public void testCall__Float() {
        System.out.println("relu float");
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        float[] expected = {1, 0, 3, 0, 0, 2, 0, 4 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
           ReLU<TFloat32> instance = new ReLU<>();
            Operand<TFloat32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat32(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    /**
     * Test of ReLU call method
     */
    @Test
    public void testCall__int() {
        System.out.println("relu int");
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = {1, 0, 3, 0, 0, 2, 0, 4 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            ReLU<TInt32> instance = new ReLU<>();
            Operand<TInt32> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }
    /**
     * Test of ReLU call method
     */
    @Test
    public void testCall__Long() {
        System.out.println("relu long");
        long[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        long[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        long[] expected = {1, 0, 3, 0, 0, 2, 0, 4 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            ReLU<TInt64> instance = new ReLU<>();
            Operand<TInt64> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt64(operand.asTensor());
            assertArrayEquals(expected, actual);
        }
    }
    
    /**
     * Test of ReLU call method
     */
    @Test
    public void testCall__Float16() {
        System.out.println("relu float16");
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        float[] expected = {1, 0, 3, 0, 0, 2, 0, 4 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            ReLU<TFloat16> instance = new ReLU<>();
            Operand<TFloat16> operand = instance.call(tf, tf.dtypes.cast(tf.constant(input), TFloat16.DTYPE));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat16(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
    /**
     * Test of ReLU call method
     */
    @Test
    public void testCall__Double() {
        System.out.println("relu double");
        double[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        double[] actual = { 0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {1, 0, 3, 0, 0, 2, 0, 4 };
        try (EagerSession session = EagerSession.create()) {
           Ops tf = Ops.create(session);
            ReLU<TFloat64> instance = new ReLU<>();
            Operand<TFloat64> operand = instance.call(tf, tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTFloat64(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }
    
}
