"""conftest.py - بيعمل ملف إكسيل تجريبي صغير (مش الملف الحقيقي) للاختبارات."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chatbot.data_loader import CARS_REQUIRED, FAULTS_REQUIRED  # noqa: E402


def _cars_rows():
    def car(cid, brand, brand_ar, model, model_ar, year, cond, price, body, cc, hp, trans, fuel,
            cons, seats, sun="لا", cam="لا", cruise="لا", leather="لا", bt="لا", notes="", check="OK"):
        return dict(Car_ID=cid, Brand=brand, Brand_AR=brand_ar, Model=model, Model_AR=model_ar, Year=year,
                    Condition=cond, Price_EGP=price, Body_Type=body, Engine_CC=cc, Horsepower=hp,
                    Transmission=trans, Fuel_Type=fuel, Fuel_Consumption_L100km=cons, Seats=seats,
                    Sunroof=sun, Rear_Camera=cam, Cruise_Control=cruise, Leather_Seats=leather,
                    Bluetooth_Screen=bt, Other_Features="", Notes=notes, Data_Check=check, Search_Text="")
    return [
        car("C1", "Toyota", "تويوتا", "Corolla", "كورولا", 2022, "جديد", 1_000_000, "Sedan", 1600, 122,
            "أوتوماتيك", "بنزين", 6.5, 5, cam="نعم", bt="نعم"),
        car("C2", "Toyota", "تويوتا", "Yaris", "ياريس", 2019, "مستعمل", 600_000, "Hatchback", 1500, 106,
            "مانيوال", "بنزين", 6.0, 5, bt="نعم"),
        car("C3", "Hyundai", "هيونداي", "Elantra", "النترا", 2023, "جديد", 1_250_000, "Sedan", 1600, 128,
            "أوتوماتيك", "بنزين", 6.8, 5, sun="نعم", cam="نعم", cruise="نعم", leather="نعم", bt="نعم"),
        car("C4", "Kia", "كيا", "Sportage", "سبورتاج", 2021, "مستعمل", 1_100_000, "SUV", 2000, 150,
            "أوتوماتيك", "بنزين", 8.5, 5, cam="نعم", bt="نعم"),
        car("C5", "Nissan", "نيسان", "Sunny", "صني", 2020, "مستعمل", 450_000, "Sedan", 1500, 99,
            "مانيوال", "بنزين", 6.2, 5),
        # صفوف لازم تتجاهل:
        car("C6", "Fiat", "فيات", "Tipo", "تيبو", 2022, "جديد", 800_000, "Sedan", 1400, 95,
            "أوتوماتيك", "بنزين", 6.0, 5, check="CHECK"),                      # Data_Check مش OK
        car("C7", "Example", "مثال", "Model", "موديل", 2020, "جديد", 1, "Sedan", 1000, 50,
            "مانيوال", "بنزين", 5.0, 5, notes="مثال للشكل فقط"),               # صف مثال
        car("C8", None, None, "NoBrand", "بلا ماركة", 2020, "جديد", 500_000, "Sedan", 1000, 50,
            "مانيوال", "بنزين", 5.0, 5),                                       # Brand فاضي
    ]


def _faults_rows():
    def fault(fid, cat, comp, symptom, kw, light, cause, sol, sev, safe, lo, hi, notes="", check="OK"):
        return dict(Fault_ID=fid, Category=cat, Component=comp, Symptom_AR=symptom, Keywords_AR=kw,
                    Warning_Light=light, Possible_Cause=cause, Solution=sol, Severity=sev,
                    Safe_To_Drive=safe, Est_Cost_EGP_Min=lo, Est_Cost_EGP_Max=hi, Applies_To="كل العربيات",
                    Notes=notes, Data_Check=check, Search_Text=f"{symptom} {kw}")
    return [
        fault("F1", "محرك", "الرادياتير", "العربية بتسخن والحرارة عالية",
              "بتسخن، الحرارة عالية، ماتور سخن، بخار طالع من الكبوت", "حرارة المحرك",
              "نقص مية الرادياتير أو عطل في الثرموستات", "وقّف العربية واطفي المحرك واتأكد من مية الرادياتير",
              "خطير", "لا", 300, 3000),
        fault("F2", "كهرباء", "البطارية", "العربية مش بتدور ومفيش صوت",
              "مش بتدور، البطارية فاضية، السلف مش شغال، ماتور ميت", "بطارية",
              "بطارية ضعيفة أو أقطاب مفكوكة", "اشحن البطارية أو بدّلها", "بسيط", "نعم", 1500, 4000),
        fault("F3", "فرامل", "تيل الفرامل", "صوت صفير عند الفرملة",
              "صفير في الفرامل، الفرامل بتزن، تيل الفرامل", "لمبة الفرامل",
              "تيل الفرامل خلص", "غيّر تيل الفرامل", "متوسط", "بحذر", 400, 1500),
        fault("F4", "تكييف", "الكمبروسر", "التكييف مش بيبرد",
              "التكييف مش بيبرد، الهوا سخن، الفريون", "", "نقص فريون", "اشحن فريون وافحص التسريب",
              "بسيط", "نعم", 300, 1200),
        fault("F5", "إطارات", "الكاوتش", "الكاوتش مفروش من ناحية واحدة",
              "كاوتش مفروش، تآكل الكاوتش، ظبط زوايا", "", "ظبط زوايا غلط", "اعمل ظبط زوايا وبدّل الكاوتش",
              "متوسط", "بحذر", 200, 6000),
        fault("F6", "محرك", "مثال", "عطل مثال", "مثال", "", "", "", "بسيط", "نعم", 1, 2,
              notes="مثال للشكل فقط"),
        fault("F7", "محرك", "غير موثّق", "عطل لسه متراجعش", "غير موثق", "", "", "", "بسيط", "نعم", 1, 2,
              check="CHECK"),
    ]


@pytest.fixture(scope="session")
def excel_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "fake_cars.xlsx"
    cars = pd.DataFrame(_cars_rows(), columns=CARS_REQUIRED)
    faults = pd.DataFrame(_faults_rows(), columns=FAULTS_REQUIRED)
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame({"README": ["ignore me"]}).to_excel(xw, sheet_name="README", index=False)
        cars.to_excel(xw, sheet_name="Cars", index=False)
        faults.to_excel(xw, sheet_name="Faults", index=False)
        pd.DataFrame({"Lists": ["x"]}).to_excel(xw, sheet_name="Lists", index=False)
    return path


@pytest.fixture(scope="session")
def data(excel_path):
    from chatbot.data_loader import load_data
    return load_data(excel_path)
