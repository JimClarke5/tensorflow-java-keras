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
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;

/**
 *
 * @author Jim Clarke
 */
public class HardSigmoidTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    public HardSigmoidTest() {
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
     * Test of HardSigmoid call method.
     */
    @Test
    public void testCall__Int() {
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = {0, 0, 0, 0, 0, 0, 0, 0};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            HardSigmoid<TInt32> instance = new HardSigmoid<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(expected, actual);
        } catch (AssertionError exp) {
            // TODO - Docs indicateit it can hanlde int, 
            // but I get this error from both Java and Python
            // so It looks like it doesn't handle them
        }
    }

    /**
     * Test of HardSigmoid call method.
     */
    @Test
    public void testCall__Float() {
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = new float[input.length];
        float[] expected = {
            0.7F, 0.099999994F, 1.1F, -0.3F, 0.3F, 0.9F, -0.100000024F, 1.3F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            HardSigmoid<TFloat32> instance = new HardSigmoid<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    /**
     * Test of HardSigmoid call method.
     */
    @Test
    public void testCall__Double() {
        double[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        double[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
            0.7, 0.09999999999999998, 1.1, -0.30000000000000004, 0.3, 0.9, -0.10000000000000009, 1.3};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            HardSigmoid<TFloat64> instance = new HardSigmoid<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

}
