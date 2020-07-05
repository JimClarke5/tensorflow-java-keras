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
public class ExponentialTest {

    private static final double EPSILON = 1e-7;
    private static final float EPSILON_F = 1e-7f;

    public ExponentialTest() {
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
     * Test of Exponential call method.
     */
    @Test
    public void testCall__Int() {
        int[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        int[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        int[] expected = {};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Exponential<TInt32> instance = new Exponential<>(tf);
            Operand<TInt32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.printTInt32(operand.asTensor());
            assertArrayEquals(expected, actual);
        } catch (AssertionError ex) {
            // expected
        }
    }

    /**
     * Test of Exponential call method.
     */
    @Test
    public void testCall__Float() {
        float[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        float[] actual = new float[input.length];
        float[] expected = {
            2.7182817F, 0.13533528F, 20.085537F, 0.01831564F, 0.36787945F, 7.389056F, 0.049787067F, 54.598152F};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Exponential<TFloat32> instance = new Exponential<>(tf);
            Operand<TFloat32> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON_F);
        }
    }

    /**
     * TTest of Exponential call method.
     */
    @Test
    public void testCall__Double() {
        double[] input = {1, -2, 3, -4, -1, 2, -3, 4};
        double[] actual = {0, 0, 0, 0, 0, 0, 0, 0};
        double[] expected = {
            2.7182818284590455, 0.1353352832366127, 20.085536923187668,
            0.018315638888734182, 0.3678794411714423, 7.38905609893065,
            0.049787068367863944, 54.598150033144236,};
        try (EagerSession session = EagerSession.create()) {
            Ops tf = Ops.create(session);
            Exponential<TFloat64> instance = new Exponential<>(tf);
            Operand<TFloat64> operand = instance.call(tf.constant(input));
            operand.asTensor().data().read(DataBuffers.of(actual));
            PrintUtils.print(operand.asTensor());
            assertArrayEquals(expected, actual, EPSILON);
        }
    }

}
