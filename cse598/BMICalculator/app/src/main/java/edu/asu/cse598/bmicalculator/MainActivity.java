package edu.asu.cse598.bmicalculator;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Color;
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
    private EditText heightEditText;
    private EditText weightEditText;
    private TextView label;
    private TextView message;
    private String uri = "";

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
        label.clearComposingText();
        String height = heightEditText.getText().toString();
        String weight = weightEditText.getText().toString();
        callApi(height, weight);
    }

    /**
     * Called when user taps educate me
     */
    public void educateMe(View view) {
        if(uri.isEmpty()){
            callApi("60", "156");
        }
        Intent browserIntent = new Intent(Intent.ACTION_VIEW, Uri.parse(uri));
        startActivity(browserIntent);
    }

    private void callApi(String height, String weight) {
        BMIApi bmiService = BMIApiClient.getClient().create(BMIApi.class);
        Call<BMIResult> call = bmiService.calculateBmi(height, weight);
        call.enqueue(new Callback<BMIResult>() {
            @Override
            public void onResponse(Call<BMIResult> call, Response<BMIResult> response) {

                String bmi = response.body().getBmi();
                List<String> more = response.body().getMore();
                String risk = response.body().getRisk();
                Toast.makeText(MainActivity.this, risk, Toast.LENGTH_SHORT).show();

                Log.i(TAG, ">>> " + bmi);
                Log.i(TAG, ">>> " + more.toString());
                Log.i(TAG, ">>> " + risk);

                label.setText(bmi);
                colourMessage(risk);
                message.setText(risk);
                uri = more.get(0);
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
