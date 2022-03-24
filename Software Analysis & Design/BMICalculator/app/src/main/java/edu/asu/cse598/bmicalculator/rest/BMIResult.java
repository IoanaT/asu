package edu.asu.cse598.bmicalculator.rest;

import java.util.List;

public class BMIResult {
    private String bmi;
    private List<String> more;
    private String risk;

    public String getBmi() {
        return bmi;
    }

    public void setBmi(String bmi) {
        this.bmi = bmi;
    }

    public List<String> getMore() {
        return more;
    }

    public void setMore(List<String> more) {
        this.more = more;
    }

    public String getRisk() {
        return risk;
    }

    public void setRisk(String risk) {
        this.risk = risk;
    }
}
