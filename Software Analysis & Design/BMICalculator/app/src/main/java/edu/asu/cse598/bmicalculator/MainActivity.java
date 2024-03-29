package edu.asu.cse598.bmicalculator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
import android.icu.text.DecimalFormat;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import java.util.List;

import edu.asu.cse598.bmicalculator.rest.BMIApi;
import edu.asu.cse598.bmicalculator.rest.BMIApiClient;
import edu.asu.cse598.bmicalculator.rest.BMIResult;
import retrofit2.Call;
import retrofit2.Callback;
import retrofit2.Response;

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "edu.asu.cse598.bmicalculator.MESSAGE";
    private static final String TAG = MainActivity.class.getSimpleName();
    private static final String DEFAULT_HEIGHT = "60";
    private static final String DEFAULT_WEIGHT = "156";
    private EditText heightEditText;
    private EditText weightEditText;
    private TextView label;
    private TextView message;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        heightEditText = (EditText) findViewById(R.id.height_input);
        weightEditText = (EditText) findViewById(R.id.weight_input);
        label = (TextView) findViewById(R.id.label);
        message = (TextView) findViewById(R.id.message);
    }

    /**
     * Called when the user taps the Send button
     */
    public void callBmi(View view) {
        String height = heightEditText.getText().toString();
        String weight = weightEditText.getText().toString();
        if  (height.isEmpty() || weight.isEmpty()){
            Toast.makeText(getApplicationContext(),"Please enter height and weight!",Toast.LENGTH_SHORT).show();
        } else {
            callApi(height, weight, false);
        }
    }

    /**
     * Called when user taps educate me
     */
    public void educateMe(View view) {
        callApi(DEFAULT_HEIGHT, DEFAULT_WEIGHT, true);
    }

    private void callApi(String height, String weight, final boolean openLink) {
        BMIApi bmiService = BMIApiClient.getClient().create(BMIApi.class);
        Call<BMIResult> call = bmiService.calculateBmi(height, weight);
        call.enqueue(new Callback<BMIResult>() {
            @Override
            public void onResponse(Call<BMIResult> call, Response<BMIResult> response) {

                String bmi = response.body().getBmi();
                List<String> more = response.body().getMore();
                String risk = response.body().getRisk();

                Log.i(TAG, ">>> " + bmi);
                Log.i(TAG, ">>> " + more.toString());
                Log.i(TAG, ">>> " + risk);

                if (openLink) {
                    Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(more.get(0)));
                    startActivity(browserIntent);
                } else {
                    label.setText(new DecimalFormat("#.##").format(Double.parseDouble(bmi)));
                    colourMessage(risk);
                    message.setText(risk);
                }
            }

            @Override
            public void onFailure(Call<BMIResult> call, Throwable t) {
                Log.e(TAG, t.toString());
            }
        });
    }

    public void colourMessage(String risk) {
        if(risk.contains("underweight")){
            message.setTextColor(Color.BLUE);
        } else if(risk.contains("normal")){
            message.setTextColor(Color.GREEN);
        } else if(risk.contains("pre-obese")){
            message.setTextColor(Color.MAGENTA);
        } else if(risk.contains("obese")){
            message.setTextColor(Color.RED);
        }
    }

}
